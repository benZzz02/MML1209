import torch
import torch.distributed as dist
import os
import time
import warnings
import argparse
import random
import json
from datetime import datetime
from statistics import mean
# [修复 1] 补全 Dict 导入
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, average_precision_score
from datetime import timedelta

from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing, InterpolationMode
from dataset import build_dataset

import ivtmetrics
from args import args
from log import logger
from loss import (
    SPLC, GRLoss, Hill, AsymmetricLossOptimized, WAN, VLPL_Loss, 
    iWAN, G_AN, LL, Weighted_Hill, Modified_VLPL,GPRLoss,BBAMLossVisual, GCELoss,SCELoss,Hill_Consistency,SPLC_Consistency
)
from utils import AverageMeter, add_weight_decay, mAP, estimate_class_distribution, run_cap_procedure, TopKCheckpointManager
from config import cfg
from consistency import ConsistencyAugmentor

from model import (
    load_clip_model, MMLSurgAdapt, Resnet, ViT, CrossModel, CLIP_for_train, VLPL, HSPNet,
    MMLSurgAdaptCoOp, MMLSurgAdaptDualCoOp, MMLSurgAdaptCoCoOp,
    MMLSurgAdaptCoOpFrozen, MMLSurgAdaptDualCoOpFrozen, MMLSurgAdaptCoCoOpFrozen, 
    CLIP_TextAttention, CLIP_TextAttentionCoOp, CLIPCoOpLoRA,
    MMLSurgAdaptSCPNet, MMLSurgAdaptSCPNet_Plus 
)
from surgvlp import SurgAVLP, CBertViT


# =============================================================================
# DDP 辅助函数
# =============================================================================
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank,timeout=timedelta(minutes=180))
        dist.barrier()
        return True, gpu
    else:
        return False, 0

# =============================================================================
# 指标计算函数
# =============================================================================
def process_cholec80(true, pred, pred_ema, video_ids, test):
    true = true[:, :7]
    pred = pred[:, :7]
    pred_ema = pred_ema[:, :7]

    unique_ids = np.unique(video_ids)
    video_f1s = {}
    video_f1s_ema = {}

    for vid in unique_ids:
        indices = np.where(video_ids == vid)[0]
        y_true_video = true[indices]
        y_pred_video = pred[indices]
        y_pred_video_ema = pred_ema[indices]

        y_pred_video = np.argmax(y_pred_video, axis=1)
        y_pred_video_ema = np.argmax(y_pred_video_ema, axis=1)
        y_true_video = np.argmax(y_true_video, axis=1)

        f1 = f1_score(y_true_video, y_pred_video, average="macro", labels=np.unique(y_true_video)) * 100
        f1_ema = f1_score(y_true_video, y_pred_video_ema, average="macro", labels=np.unique(y_true_video)) * 100

        video_f1s[vid] = f1
        video_f1s_ema[vid] = f1_ema

    f1_score_reg = mean(video_f1s.values())
    f1_score_ema = mean(video_f1s_ema.values())

    if test:
        return f1_score_reg, None, video_f1s, None

    return f1_score_reg, f1_score_ema, video_f1s, video_f1s_ema

def process_endo(true, pred, pred_ema, video_ids, test):
    true = true[:, 7:10]
    pred = pred[:, 7:10]
    pred_ema = pred_ema[:, 7:10]

    num_classes = true.shape[1]
    per_class_map = {}
    per_class_map_ema = {}

    for label in range(num_classes):
        avg_precision = average_precision_score(true[:, label], pred[:, label])
        avg_precision_ema = average_precision_score(true[:, label], pred_ema[:, label])
        
        per_class_map[f"C{label}"] = avg_precision * 100.0
        per_class_map_ema[f"C{label}"] = avg_precision_ema * 100.0

    mean_ap = mean(per_class_map.values())
    mean_ap_ema = mean(per_class_map_ema.values())

    if test:
        return mean_ap, None, per_class_map, None

    return mean_ap, mean_ap_ema, per_class_map, per_class_map_ema

def resolve_nan(classwise):
    classwise[classwise == -0.0] = np.nan
    return classwise

def process_cholect50(true, pred, pred_ema, video_ids, test):
    true = true[:, 10:]
    pred = pred[:, 10:]
    pred_ema = pred_ema[:, 10:]
    unique_vids = np.unique(video_ids)

    ap_i_list, ap_v_list, ap_t_list = [], [], []
    ap_iv_list, ap_it_list, ap_ivt_list = [], [], []

    ap_i_list_ema, ap_v_list_ema, ap_t_list_ema = [], [], []
    ap_iv_list_ema, ap_it_list_ema, ap_ivt_list_ema = [], [], []

    for vid in unique_vids:
        indices = np.where(video_ids == vid)[0]
        ivt_labels = true[indices]
        ivt_preds = pred[indices]
        ivt_preds_ema = pred_ema[indices]

        filter_obj = ivtmetrics.Disentangle()

        # Extract components
        i_labels = filter_obj.extract(inputs=ivt_labels, component="i")
        v_labels = filter_obj.extract(inputs=ivt_labels, component="v")
        t_labels = filter_obj.extract(inputs=ivt_labels, component="t")
        iv_labels = filter_obj.extract(inputs=ivt_labels, component="iv")
        it_labels = filter_obj.extract(inputs=ivt_labels, component="it")

        i_preds = filter_obj.extract(inputs=ivt_preds, component="i")
        v_preds = filter_obj.extract(inputs=ivt_preds, component="v")
        t_preds = filter_obj.extract(inputs=ivt_preds, component="t")
        iv_preds = filter_obj.extract(inputs=ivt_preds, component="iv")
        it_preds = filter_obj.extract(inputs=ivt_preds, component="it")

        i_preds_ema = filter_obj.extract(inputs=ivt_preds_ema, component="i")
        v_preds_ema = filter_obj.extract(inputs=ivt_preds_ema, component="v")
        t_preds_ema = filter_obj.extract(inputs=ivt_preds_ema, component="t")
        iv_preds_ema = filter_obj.extract(inputs=ivt_preds_ema, component="iv")
        it_preds_ema = filter_obj.extract(inputs=ivt_preds_ema, component="it")

        # Calculate APs
        ap_i = average_precision_score(i_labels, i_preds, average=None) * 100
        ap_v = average_precision_score(v_labels, v_preds, average=None) * 100
        ap_t = average_precision_score(t_labels, t_preds, average=None) * 100
        ap_iv = average_precision_score(iv_labels, iv_preds, average=None) * 100
        ap_it = average_precision_score(it_labels, it_preds, average=None) * 100
        ap_ivt = average_precision_score(ivt_labels, ivt_preds, average=None) * 100

        ap_i_ema = average_precision_score(i_labels, i_preds_ema, average=None) * 100
        ap_v_ema = average_precision_score(v_labels, v_preds_ema, average=None) * 100
        ap_t_ema = average_precision_score(t_labels, t_preds_ema, average=None) * 100
        ap_iv_ema = average_precision_score(iv_labels, iv_preds_ema, average=None) * 100
        ap_it_ema = average_precision_score(it_labels, it_preds_ema, average=None) * 100
        ap_ivt_ema = average_precision_score(ivt_labels, ivt_preds_ema, average=None) * 100

        # Resolve NaNs and Append
        ap_list_map = [
            (ap_i, ap_i_list), (ap_v, ap_v_list), (ap_t, ap_t_list),
            (ap_iv, ap_iv_list), (ap_it, ap_it_list), (ap_ivt, ap_ivt_list)
        ]
        for val, lst in ap_list_map:
            lst.append(resolve_nan(val).reshape([1, -1]))

        ap_list_map_ema = [
            (ap_i_ema, ap_i_list_ema), (ap_v_ema, ap_v_list_ema), (ap_t_ema, ap_t_list_ema),
            (ap_iv_ema, ap_iv_list_ema), (ap_it_ema, ap_it_list_ema), (ap_ivt_ema, ap_ivt_list_ema)
        ]
        for val, lst in ap_list_map_ema:
            lst.append(resolve_nan(val).reshape([1, -1]))

    # Aggregate Metrics
    def get_mean_ap(lst):
        if len(lst) == 0: return 0.0
        concatenated = np.concatenate(lst, axis=0)
        return np.nanmean(np.nanmean(concatenated, axis=0))

    aps = {
        "AP_i": get_mean_ap(ap_i_list),
        "AP_v": get_mean_ap(ap_v_list),
        "AP_t": get_mean_ap(ap_t_list),
        "AP_iv": get_mean_ap(ap_iv_list),
        "AP_it": get_mean_ap(ap_it_list),
        "AP_ivt": get_mean_ap(ap_ivt_list)
    }

    aps_ema = {
        "AP_i": get_mean_ap(ap_i_list_ema),
        "AP_v": get_mean_ap(ap_v_list_ema),
        "AP_t": get_mean_ap(ap_t_list_ema),
        "AP_iv": get_mean_ap(ap_iv_list_ema),
        "AP_it": get_mean_ap(ap_it_list_ema),
        "AP_ivt": get_mean_ap(ap_ivt_list_ema)
    }

    if test:
        return aps, None

    return aps, aps_ema

def save_results(a, b, c, test, dir, method):
    if not is_main_process(): return

    cholec80_data = {"F1_score": a[0], "F1_score_EMA": a[1], "Per_video_f1": a[2], "Per_video_f1_EMA": a[3]}
    endo_data = {"mAP": b[0], "mAP_EMA": b[1], "mAP_per_class": b[2], "mAP_per_class_EMA": b[3]}
    cholect50_data = {"AP": c[0], "AP_EMA": c[1]}

    data = {
        "Test": test,
        "Cholec80": cholec80_data,
        "Endoscapes": endo_data,
        "CholecT50": cholect50_data
    }

    folder_name = f"results/{dir}"
    os.makedirs(f"results/{dir}", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{folder_name}/result_{timestamp}_{method}_test.json" if test else f"{folder_name}/result_{timestamp}.json"

    try:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print("Results saved successfully")
    except Exception as e:
        print(f"Error in saving results : {e}")

def calculate_metrics(labels, preds, preds_ema, video_ids, test, dir, method=None):
    if not is_main_process(): return None, None, None

    labels = np.round(labels)
    datasets = ["cholec80", "endoscapes", "cholect50"]
    a, b, c = None, None, None

    for dataset in datasets:
        mask = np.array([dataset in v for v in video_ids])

        if not np.any(mask):
            continue

        filtered_labels = labels[mask]
        filtered_preds = preds[mask]
        filtered_preds_ema = preds_ema[mask]
        filtered_video_ids = video_ids[mask]

        if dataset == "cholec80":
            a = process_cholec80(filtered_labels, filtered_preds, filtered_preds_ema, filtered_video_ids, test)
        elif dataset == "endoscapes":
            b = process_endo(filtered_labels, filtered_preds, filtered_preds_ema, filtered_video_ids, test)
        elif dataset == "cholect50":
            c = process_cholect50(filtered_labels, filtered_preds, filtered_preds_ema, filtered_video_ids, test)

    if a and b and c:
        save_results(a, b, c, test, dir, method)
    else:
        pass
    
    print("Calculating metrics done")
    return a, b, c

def save_best(trainer, if_ema_better: bool, dir, method) -> None:
    if not is_main_process(): return

    state_dict = trainer.model.module.state_dict() if hasattr(trainer.model, "module") else trainer.model.state_dict()
    ema_state_dict = trainer.ema.module.state_dict() if hasattr(trainer.ema, "module") else trainer.ema.state_dict()

    save_path = os.path.join(cfg.checkpoint, f'{dir}/{method}')
    os.makedirs(save_path, exist_ok=True)

    if if_ema_better:
        torch.save(ema_state_dict, os.path.join(save_path, 'model-highest.ckpt'))
    else:
        torch.save(state_dict, os.path.join(save_path, 'model-highest.ckpt'))
    
    torch.save(state_dict, os.path.join(save_path, 'model-highest-regular.ckpt'))
    torch.save(ema_state_dict, os.path.join(save_path, 'model-highest-ema.ckpt'))

# =============================================================================
# SCPNetTrainer 类
# =============================================================================
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import ModelEma, get_ema_co

class SCPNetTrainer():
    def __init__(self, distributed=False, gpu_id=0) -> None:
        super().__init__()
        self.distributed = distributed
        self.gpu_id = gpu_id

        clip_model, _ = load_clip_model()
        image_size = cfg.image_size

        train_preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        val_preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        if cfg.perform_init:
            train_loader, val_loader, sp_loader, _, init_train_loader, init_val_loader = build_dataset(train_preprocess, val_preprocess, distributed=distributed)
        else:
            train_loader, val_loader, sp_loader, _ = build_dataset(train_preprocess, val_preprocess, distributed=distributed)

        if cfg.perform_init:
            _, _, _, test_loader, _, _ = build_dataset(train_preprocess, val_preprocess, distributed=False)
        else:
            _, _, _, test_loader = build_dataset(train_preprocess, val_preprocess, distributed=False)
            
        logger.info("Creating a non-distributed validation loader for accurate evaluation...")
        if cfg.perform_init:
            _, clean_val_loader, clean_sp_loader, _, _, _ = build_dataset(train_preprocess, val_preprocess, distributed=False)
        else:
            _, clean_val_loader, clean_sp_loader, _ = build_dataset(train_preprocess, val_preprocess, distributed=False)
        
        self.clean_val_loader = clean_val_loader
        if cfg.val_sp: self.clean_sp_loader = clean_sp_loader

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if cfg.val_sp: self.val_sp_loader = sp_loader
        if cfg.perform_init: self.init_train_loader, self.init_val_loader = init_train_loader, init_val_loader
        classnames = val_loader.dataset.labels()

        # 模型初始化
        model_name = cfg.model
        if cfg.backbone == 'SurgVLP':
            self.model = SurgAVLP(clip_model, classnames, cfg.bert_path, cfg.vlp_weights)
        elif model_name == 'SurgAdapt': self.model = MMLSurgAdapt(classnames, clip_model)
        elif model_name == 'HSPNet': self.model = HSPNet(classnames, clip_model)
        elif model_name == 'VLPL': self.model = VLPL(classnames, clip_model)
        elif model_name == 'Resnet': self.model = Resnet(classnames, clip_model)
        elif model_name == 'ViT': self.model = ViT(classnames, clip_model)
        elif model_name == 'CLIP': self.model = CLIP_for_train(classnames, clip_model)
        elif model_name == 'CrossModel': self.model = CrossModel(classnames, clip_model)
        elif model_name == 'SurgAdapt-CoOp': self.model = MMLSurgAdaptCoOp(classnames, clip_model)
        elif model_name == 'SurgAdapt-DualCoOp': self.model = MMLSurgAdaptDualCoOp(classnames, clip_model)
        elif model_name == 'SurgAdapt-CoCoOp': self.model = MMLSurgAdaptCoCoOp(classnames, clip_model)
        elif model_name == 'SurgAdapt-CoOp-Frozen': self.model = MMLSurgAdaptCoOpFrozen(classnames, clip_model)
        elif model_name == 'SurgAdapt-DualCoOp-Frozen': self.model = MMLSurgAdaptDualCoOpFrozen(classnames, clip_model)
        elif model_name == 'SurgAdapt-CoCoOp-Frozen': self.model = MMLSurgAdaptCoCoOpFrozen(classnames, clip_model)
        elif model_name == 'CLIP-TextAttention': self.model = CLIP_TextAttention(classnames, clip_model)
        elif model_name == 'CLIP-TextAttention-CoOp': self.model = CLIP_TextAttentionCoOp(classnames, clip_model)
        elif model_name == 'CLIP-CoOp-LoRA': self.model = CLIPCoOpLoRA(classnames, clip_model)
        # [SCPNet]
        elif model_name == 'SCPNet': self.model = MMLSurgAdaptSCPNet(classnames, clip_model)
        elif model_name == 'SCPNet_Plus': self.model = MMLSurgAdaptSCPNet_Plus(classnames, clip_model)
        else:
            raise NameError(f"Model '{model_name}' not recognized.")

        logger.info(f"Successfully initialized model: {model_name}")
        self.classnames = classnames

        self.model.cuda(self.gpu_id)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id, find_unused_parameters=True)

        ema_co = get_ema_co()
        logger.info(f"EMA CO: {ema_co}")
        self.model_unwrap = self.model.module if self.distributed else self.model
        self.ema = ModelEma(self.model_unwrap, ema_co)
        
        # 一致性增强配置
        self.use_consistency = getattr(cfg, 'use_consistency', False)
        if self.use_consistency:
            self.augmentor = ConsistencyAugmentor(
                flip_prob=getattr(cfg, 'cons_flip_prob', 0.5),
                brightness_range=getattr(cfg, 'cons_brightness_range', 0.4),
                brightness_min=getattr(cfg, 'cons_brightness_min', 0.8),
                gray_prob=getattr(cfg, 'cons_gray_prob', 0.2)
            )
            logger.info("Consistency regularization is ENABLED")
        else:
            self.augmentor = None
            logger.info("Consistency regularization is DISABLED")
    
    # [核心修复] train 函数中的 Loss 调用逻辑
    def train(self, input, target, criterion, epoch, epoch_i) -> torch.Tensor:
        image = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        need_consistency = (
            self.use_consistency and
            self.model.training and
            self.augmentor is not None
        )
        
        image_strong = None
        if need_consistency:
            image_strong = self.augmentor.augment_batch(image.cpu()).cuda(non_blocking=True)

        if getattr(criterion, 'needs_features', False):
            # BBAM etc
            with autocast():
                features = self.model_unwrap.image_encoder(image.type(self.model_unwrap.dtype))
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            loss, _ = criterion(features, target, epoch)
            
        else:
            with autocast():
                # [适配] SCPNet 系列：支持 (image, strong) 输入和多返回值
                if cfg.model in ['SCPNet', 'SCPNet_Plus', 'MMLSurgAdaptSCPNet_Plus']:
                    output_tuple = self.model(image, image_strong)
                    
                    if isinstance(output_tuple, tuple):
                        logits, A_star, loss_cons_internal = output_tuple
                    else:
                        logits, A_star, loss_cons_internal = output_tuple, None, 0.0
                    
                    # [修复 TypeError] 
                    # 尝试传递 A_star，如果 Loss 不支持（如标准 Hill），则回退到普通调用
                    try:
                        loss_main, _ = criterion(logits, target, epoch, A_star=A_star)
                    except TypeError:
                        loss_main, _ = criterion(logits, target, epoch)
                    
                    w_cons = getattr(cfg, 'w_cons', 1.0)
                    loss = loss_main + w_cons * loss_cons_internal

                # [适配] 旧模型逻辑
                else:
                    if need_consistency:
                        combined = torch.cat([image, image_strong], dim=0)
                        combined_output = self.model(combined).float()
                        
                        batch_size = image.shape[0]
                        output_clean = combined_output[:batch_size]
                        output_aug = combined_output[batch_size:]
                        
                        if hasattr(criterion, 'cons_weight'):
                            loss, _ = criterion(output_clean, output_aug, target, epoch)
                        else:
                            loss, _ = criterion(output_clean, target, epoch)
                    else:
                        output = self.model(image).float()
                        if hasattr(criterion, 'cons_weight'):
                            loss, _ = criterion(output, None, target, epoch)
                        else:
                            loss, _ = criterion(output, target, epoch)
            
        return loss

# =============================================================================
# 验证、初始化验证、保存函数
# =============================================================================
def validate(trainer, epoch: int, dir, criterion=None) -> dict:
    if not is_main_process():
        if dist.is_initialized(): dist.barrier()
        return {}

    trainer.model.eval()
    logger.info("Start validation on single GPU (rank 0)...")

    loss_dict = {
        'SPLC': SPLC, 'GRLoss': GRLoss, 'Hill': Hill,
        'BCE': lambda: AsymmetricLossOptimized(gamma_neg=0, gamma_pos=0, clip=0),
        'Focal': lambda: AsymmetricLossOptimized(gamma_neg=2, gamma_pos=2, clip=0),
        'ASL': lambda: AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05),
        'WAN': WAN, 'VLPL_Loss': VLPL_Loss, 'Modified_VLPL': Modified_VLPL, 'iWAN': iWAN,
        'G-AN': G_AN, 'LL-R': lambda: LL(scheme='LL-R'), 'LL-Ct': LL,
        'Weighted_Hill': Weighted_Hill,'GPRLoss': GPRLoss , 'BBAM': lambda: BBAMLossVisual(num_classes=cfg.num_classes, s=10.0, m=0.4, start_epoch=5),
        'GCE': lambda: GCELoss(q=0.7),
        'SCE': lambda: SCELoss(alpha=1.0, beta=1.0),
        'Hill_Consistency': lambda: Hill_Consistency(
            lamb=getattr(cfg, 'lamb', 1.5), margin=getattr(cfg, 'margin', 1.0), gamma=getattr(cfg, 'gamma', 2.0), cons_weight=getattr(cfg, 'cons_weight', 20.0), cons_temp=getattr(cfg, 'cons_temp', 1.0)
        ),
        'SPLC_Consistency': lambda: SPLC_Consistency(
            tau=getattr(cfg, 'tau', 0.6), change_epoch=getattr(cfg, 'change_epoch', 1), margin=getattr(cfg, 'margin', 1.0), gamma=getattr(cfg, 'gamma', 2.0), cons_weight=getattr(cfg, 'cons_weight', 20.0), cons_temp=getattr(cfg, 'cons_temp', 1.0)
        ),
    }
    if criterion is None:
        criterion = loss_dict.get(cfg.loss, lambda: None)()
        if torch.cuda.is_available() and criterion is not None: criterion = criterion.cuda()
    
    sigmoid = torch.sigmoid
    preds_regular, preds_ema, all_vids, targets = [], [], [], []
    losses, losses_ema = [], []

    model_to_run = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    ema_to_run = trainer.ema.module if hasattr(trainer.ema, 'module') else trainer.ema

    for _, (input, target, vid) in enumerate(trainer.val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            with autocast():
                # [适配] 取第一个返回值作为 Logits
                out = model_to_run(input)
                if isinstance(out, tuple): output_logits = out[0]
                else: output_logits = out
                
                out_ema = ema_to_run(input)
                if isinstance(out_ema, tuple): output_ema_logits = out_ema[0]
                else: output_ema_logits = out_ema

                output_regular = sigmoid(output_logits)
                output_ema = sigmoid(output_ema_logits)

        loss, _ = criterion(output_logits, target, epoch)
        loss_ema, _ = criterion(output_ema_logits, target, epoch)

        losses.append(loss.item())
        losses_ema.append(loss_ema.item())
        
        preds_regular.append(output_regular.cpu().numpy())
        preds_ema.append(output_ema.cpu().numpy())
        targets.append(target.cpu().numpy())
        all_vids.extend(vid)

    loss_mean = mean(losses)
    loss_mean_ema = mean(losses_ema)

    all_labels = np.concatenate(targets, axis=0)
    all_predictions_reg = np.concatenate(preds_regular, axis=0)
    all_predictions_ema = np.concatenate(preds_ema, axis=0)
    all_vids = np.array(all_vids)

    calculate_metrics(all_labels, all_predictions_reg, all_predictions_ema, all_vids, False, dir)

    mAP_score_regular = mAP(all_labels, all_predictions_reg)
    mAP_score_ema = mAP(all_labels, all_predictions_ema)
    logger.info(f"mAP score regular {mAP_score_regular:.2f}, mAP score EMA {mAP_score_ema:.2f}")
    
    mAP_max = max(mAP_score_regular, mAP_score_ema)
    if_ema_better_mAP = mAP_score_ema >= mAP_score_regular

    logger.info(f"Loss on pp regular {loss_mean:.2f}, Loss on pp EMA {loss_mean_ema:.2f}")
    loss_min = min(loss_mean, loss_mean_ema)
    if_ema_better_loss_pp = loss_mean_ema <= loss_mean

    evals = {
        'pp_map': mAP_max,
        'pp_map_if_better': if_ema_better_mAP,
        'pp_loss': loss_min,
        'pp_loss_if_better': if_ema_better_loss_pp
    }

    if cfg.val_sp:
        logger.info("Start sp validation...")
        losses_sp, losses_ema_sp = [], []
        for _, (input, target, vid) in enumerate(trainer.val_sp_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            with torch.no_grad():
                with autocast():
                    out = model_to_run(input)
                    if isinstance(out, tuple): output_logits = out[0]
                    else: output_logits = out
                    
                    out_ema = ema_to_run(input)
                    if isinstance(out_ema, tuple): output_ema_logits = out_ema[0]
                    else: output_ema_logits = out_ema

            loss, _ = criterion(output_logits, target, epoch)
            loss_ema, _ = criterion(output_ema_logits, target, epoch)
            losses_sp.append(loss.item())
            losses_ema_sp.append(loss_ema.item())
        
        loss_mean_sp = mean(losses_sp)
        loss_mean_ema_sp = mean(losses_ema_sp)
        
        logger.info(f"Loss on sp regular {loss_mean_sp:.2f}, Loss on sp EMA {loss_mean_ema_sp:.2f}")
        loss_min_sp = min(loss_mean_sp, loss_mean_ema_sp)
        if_ema_better_loss_sp = loss_mean_ema_sp <= loss_mean_sp
        
        evals['sp_loss'] = loss_min_sp
        evals['sp_loss_if_better'] = if_ema_better_loss_sp

    if dist.is_initialized(): dist.barrier()
    return evals

def init_validate(trainer, epoch: int):
    if not is_main_process():
        if dist.is_initialized(): dist.barrier()
        return float('inf'), False

    trainer.model.eval()
    logger.info("Start init validation on single GPU (rank 0)...")
    criterion = nn.BCEWithLogitsLoss()
    losses, losses_ema = [], []

    model_to_run = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    ema_to_run = trainer.ema.module if hasattr(trainer.ema, 'module') else trainer.ema

    for _, (input, target) in enumerate(trainer.init_val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            with autocast():
                out = model_to_run(input)
                if isinstance(out, tuple): output_logits = out[0]
                else: output_logits = out
                
                out_ema = ema_to_run(input)
                if isinstance(out_ema, tuple): output_ema_logits = out_ema[0]
                else: output_ema_logits = out_ema

        loss = criterion(output_logits, target)
        loss_ema = criterion(output_ema_logits, target)
        losses.append(loss.item())
        losses_ema.append(loss_ema.item())

    loss_mean = mean(losses)
    loss_mean_ema = mean(losses_ema)
    
    logger.info(f"Loss on init val {loss_mean:.2f}, Loss on init val EMA {loss_mean_ema:.2f}")
    if_ema_better = loss_mean_ema <= loss_mean
    logger.info('Validation complete')

    loss_min = min(loss_mean, loss_mean_ema)
    if dist.is_initialized(): dist.barrier()
    return loss_min, if_ema_better

def save_best_init(trainer, if_ema_better, dir):
    if not is_main_process(): return
    state_dict = trainer.model.module.state_dict() if hasattr(trainer.model, "module") else trainer.model.state_dict()
    ema_state_dict = trainer.ema.module.state_dict() if hasattr(trainer.ema, "module") else trainer.ema.state_dict()
    
    save_path = os.path.join(cfg.checkpoint, f'{dir}/init')
    os.makedirs(save_path, exist_ok=True)
    
    if if_ema_better:
        torch.save(ema_state_dict, os.path.join(save_path, 'model-highest.ckpt'))
    else:
        torch.save(state_dict, os.path.join(save_path, 'model-highest.ckpt'))

def train(trainer, dir) -> list:
    loss_dict = {
        'SPLC': SPLC, 'GRLoss': GRLoss, 'Hill': Hill,
        'BCE': lambda: AsymmetricLossOptimized(gamma_neg=0, gamma_pos=0, clip=0),
        'Focal': lambda: AsymmetricLossOptimized(gamma_neg=2, gamma_pos=2, clip=0),
        'ASL': lambda: AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05),
        'WAN': WAN, 'VLPL_Loss': VLPL_Loss, 'Modified_VLPL': Modified_VLPL, 'iWAN': iWAN,
        'G-AN': G_AN, 'LL-R': lambda: LL(scheme='LL-R'), 'LL-Ct': LL, 'Weighted_Hill': Weighted_Hill, 'GPRLoss': GPRLoss,
        'BBAM': lambda: BBAMLossVisual(num_classes=cfg.num_classes, s=10.0, m=0.4, start_epoch=5),
        'GCE': lambda: GCELoss(q=0.7),
        'SCE': lambda: SCELoss(alpha=1.0, beta=1.0),
        'Hill_Consistency': lambda: Hill_Consistency(lamb=getattr(cfg, 'lamb', 1.5), margin=getattr(cfg, 'margin', 1.0), gamma=getattr(cfg, 'gamma', 2.0), cons_weight=getattr(cfg, 'cons_weight', 20.0), cons_temp=getattr(cfg, 'cons_temp', 1.0)),
        'SPLC_Consistency': lambda: SPLC_Consistency(tau=getattr(cfg, 'tau', 0.6), change_epoch=getattr(cfg, 'change_epoch', 1), margin=getattr(cfg, 'margin', 1.0), gamma=getattr(cfg, 'gamma', 2.0), cons_weight=getattr(cfg, 'cons_weight', 20.0), cons_temp=getattr(cfg, 'cons_temp', 1.0)),
    }
    criterion = loss_dict.get(cfg.loss, lambda: None)()
    if criterion is None: raise ValueError(f"Loss function '{cfg.loss}' not found.")
    if torch.cuda.is_available(): criterion = criterion.cuda()
    if is_main_process(): print(f"Using criterion: {criterion}")
    
    parameters = add_weight_decay(trainer.model, cfg.weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=cfg.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    steps_per_epoch = len(trainer.train_loader)
    accumulation_steps = getattr(cfg, 'accumulation_steps', 1)

    top_k_num = getattr(cfg, 'top_k', 3)
    best_map_list = []
    best_pp_loss_list = []
    best_sp_loss_list = []
    
    save_dir_base = os.path.join(cfg.checkpoint, dir)
    if is_main_process(): os.makedirs(save_dir_base, exist_ok=True)

    use_cap = getattr(cfg, 'use_cap', False)
    cap_start_epoch = getattr(cfg, 'cap_start_epoch', 5)
    cap_ratio = getattr(cfg, 'cap_ratio', 0.6)
    pos_freq = None
    if use_cap:
        if is_main_process():
            pos_freq = estimate_class_distribution(trainer.train_loader.dataset, cfg.num_classes)
            if dist.is_initialized() and pos_freq is not None:
                pos_freq_t = torch.from_numpy(pos_freq).cuda()
                dist.broadcast(pos_freq_t, 0)
        else:
            if dist.is_initialized():
                pos_freq_t = torch.zeros(cfg.num_classes).cuda()
                dist.broadcast(pos_freq_t, 0)
                pos_freq = pos_freq_t.cpu().numpy()
    
    trainer.model.train()

    if cfg.perform_init:
        init_optimizer = torch.optim.Adam(params=parameters, lr=cfg.init_lr, weight_decay=0)
        min_init_loss = float('inf')
        best_epoch_init = 0
        init_steps_per_epoch = len(trainer.init_train_loader)
        
        for epoch in range(cfg.init_epochs):
            if hasattr(trainer.init_train_loader, 'sampler') and hasattr(trainer.init_train_loader.sampler, 'set_epoch'):
                 trainer.init_train_loader.sampler.set_epoch(epoch)

            for i, (input, target) in enumerate(trainer.init_train_loader):
                init_optimizer.zero_grad()
                target = target.cuda(non_blocking=True)
                image = input.cuda(non_blocking=True)
                with autocast():
                    out = trainer.model(image)
                    if isinstance(out, tuple): out = out[0]
                    output = out.float()
                loss = nn.BCEWithLogitsLoss()(output, target)
                scaler.scale(loss).backward()
                scaler.step(init_optimizer)
                scaler.update()
                trainer.ema.update(trainer.model)

                if i % 100 == 0 and is_main_process():
                    logger.info('Init Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.4f}'
                        .format(epoch, cfg.init_epochs, str(i).zfill(3), str(init_steps_per_epoch).zfill(3), cfg.init_lr, loss.item()))
            
            min_loss, is_ema_better = init_validate(trainer, epoch)

            if is_main_process():
                if min_loss < min_init_loss:
                    min_init_loss = min_loss
                    best_epoch_init = epoch
                    save_best_init(trainer, is_ema_better, dir)
                logger.info('current_init_loss = {:.2f}, min_init_loss = {:.2f}, best_epoch={}, is_ema_better={}\n'.
                    format(min_loss, min_init_loss, best_epoch_init, is_ema_better))
            
            trainer.model.train()
        
        if dist.is_initialized(): dist.barrier()
        if is_main_process():
            map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.gpu_id}
            path = f"{cfg.checkpoint}/{dir}/init/model-highest.ckpt"
            if os.path.exists(path):
                state_dict = torch.load(path, map_location=map_location)
                model_to_load = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                model_to_load.load_state_dict(state_dict, strict=True)
                logger.info("Best init model loaded by main process.")
        if dist.is_initialized(): dist.barrier()
        trainer.model.train()

    if is_main_process(): logger.info(f"Gradient Accumulation Steps: {accumulation_steps}")

    for epoch in range(cfg.epochs):
        if hasattr(trainer.train_loader, 'sampler') and hasattr(trainer.train_loader.sampler, 'set_epoch'):
             trainer.train_loader.sampler.set_epoch(epoch)

        if use_cap and epoch >= cap_start_epoch and pos_freq is not None:
             if dist.is_initialized(): dist.barrier()
             run_cap_procedure(trainer, trainer.train_loader, pos_freq, device=torch.device(f"cuda:{cfg.gpu_id}"), ratio=cap_ratio)
             if dist.is_initialized(): dist.barrier()
    
        optimizer.zero_grad()
        for i, batch_data in enumerate(trainer.train_loader):
            input = batch_data[0]
            target = batch_data[1]
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            
            loss = trainer.train(input, target, criterion, epoch, i)
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                trainer.ema.update(trainer.model)
            
            if i % 100 == 0 and is_main_process():
                log_loss = loss.item() * accumulation_steps
                logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.4f}'
                    .format(epoch, cfg.epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3), cfg.lr, log_loss))

        evals = validate(trainer, epoch, dir, criterion=criterion)

        if is_main_process() and evals:
            cur_map = evals['pp_map']
            cur_pp_loss = evals['pp_loss']
            
            use_ema = evals.get('pp_map_if_better', False)
            if use_ema and hasattr(trainer, 'ema'):
                state_dict = trainer.ema.module.state_dict() if hasattr(trainer.ema, "module") else trainer.ema.state_dict()
                suffix = "ema"
            else:
                state_dict = trainer.model.module.state_dict() if hasattr(trainer.model, "module") else trainer.model.state_dict()
                suffix = "reg"
            
            filename = f"epoch_{epoch}_mAP_{cur_map:.2f}_loss_{cur_pp_loss:.4f}_{suffix}.ckpt"
            filepath = os.path.join(save_dir_base, filename)
            
            save_needed = False
            best_map_list.append((cur_map, epoch, filepath))
            best_map_list.sort(key=lambda x: x[0], reverse=True)
            if any(x[1] == epoch for x in best_map_list[:top_k_num]): save_needed = True

            best_pp_loss_list.append((cur_pp_loss, epoch, filepath))
            best_pp_loss_list.sort(key=lambda x: x[0])
            if any(x[1] == epoch for x in best_pp_loss_list[:top_k_num]): save_needed = True
                
            if cfg.val_sp and 'sp_loss' in evals:
                best_sp_loss_list.append((evals['sp_loss'], epoch, filepath))
                best_sp_loss_list.sort(key=lambda x: x[0])
                if any(x[1] == epoch for x in best_sp_loss_list[:top_k_num]): save_needed = True
            
            if save_needed:
                torch.save(state_dict, filepath)
                logger.info(f"Saved Top-K model: {filename}")

            valid_paths = set()
            valid_paths.update([x[2] for x in best_map_list[:top_k_num]])
            valid_paths.update([x[2] for x in best_pp_loss_list[:top_k_num]])
            if cfg.val_sp: valid_paths.update([x[2] for x in best_sp_loss_list[:top_k_num]])
            
            for f in os.listdir(save_dir_base):
                if f.endswith('.ckpt') and "epoch_" in f:
                    full_p = os.path.join(save_dir_base, f)
                    if full_p not in valid_paths:
                        try:
                            os.remove(full_p)
                            logger.info(f"Removed worse model: {f}")
                        except: pass
            
            best_map_list = best_map_list[:top_k_num]
            best_pp_loss_list = best_pp_loss_list[:top_k_num]
            if cfg.val_sp: best_sp_loss_list = best_sp_loss_list[:top_k_num]

        trainer.model.train()
        scheduler.step()

    final_paths = []
    if is_main_process():
        if best_map_list: final_paths.extend([x[2] for x in best_map_list])
        if best_pp_loss_list: final_paths.extend([x[2] for x in best_pp_loss_list])
        if best_sp_loss_list: final_paths.extend([x[2] for x in best_sp_loss_list])
        final_paths = list(set(final_paths))
        
    return final_paths

def test(trainer, dir, checkpoint_paths=None) -> None:
    if not is_main_process(): return
    logger.info("Starting test phase...")
    target_paths = checkpoint_paths if checkpoint_paths else []
    
    if not target_paths and cfg.test:
        import glob
        base = os.path.join(cfg.checkpoint, dir)
        target_paths = glob.glob(os.path.join(base, "*.ckpt"))
        
    if not target_paths:
        logger.warning("No checkpoints provided to test.")
        return

    target_paths.sort()
    sigmoid = torch.sigmoid
    processed_epochs = set()

    for ckpt_path in target_paths:
        if not os.path.exists(ckpt_path): continue
        filename = os.path.basename(ckpt_path)
        epoch_id = "unknown"
        try:
            parts = filename.split('_')
            if parts[0] == 'epoch': epoch_id = parts[1]
        except: pass
        
        if epoch_id != "unknown" and epoch_id in processed_epochs:
            logger.info(f"Skipping {filename} (Epoch {epoch_id} already tested)")
            continue
            
        logger.info(f"Testing checkpoint: {filename} ...")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.gpu_id}
        state_dict = torch.load(ckpt_path, map_location=map_location)
        model_to_run = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        model_to_run.load_state_dict(state_dict, strict=True)
        model_to_run.eval()

        preds, targets, all_vids = [], [], []
        for i, (input, target, vid) in enumerate(trainer.test_loader):
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            with torch.no_grad():
                out = model_to_run(input)
                # [适配] 测试时也要处理 tuple 返回值
                if isinstance(out, tuple): output_logits = out[0]
                else: output_logits = out
                output = sigmoid(output_logits)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            all_vids.extend(vid)

        all_labels = np.concatenate(targets, axis=0)
        all_predictions = np.concatenate(preds, axis=0)
        all_vids_np = np.array(all_vids)
        method_name = filename.replace('.ckpt', '')
        calculate_metrics(all_labels, all_predictions, all_predictions, all_vids_np, True, dir, method=method_name)
        processed_epochs.add(epoch_id)
        logger.info(f"Finished testing {method_name}")

def makedir(dir):
    if not is_main_process(): return
    os.makedirs(f"{cfg.checkpoint}/{dir}", exist_ok=True)
    os.makedirs(f"results/{dir}", exist_ok=True)
    for method in cfg.val_methods:
        os.makedirs(f"{cfg.checkpoint}/{dir}/{method}", exist_ok=True)
    os.makedirs(f"{cfg.checkpoint}/{dir}/init", exist_ok=True)

def main():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    warnings.filterwarnings("ignore", category=UserWarning) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    
    is_distributed, gpu_id = setup_distributed()
    cfg.distributed = is_distributed
    cfg.gpu_id = gpu_id

    s = cfg.seed + (gpu_id if is_distributed else 0)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    random.seed(s)
    np.random.seed(s)
    
    if is_main_process():
        logger.info(f'Seed {s}, DDP: {is_distributed}, GPU: {gpu_id}')
    
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    
    dir = cfg.dir
    makedir(dir)
    if is_distributed: dist.barrier()

    trainer = SCPNetTrainer(distributed=is_distributed, gpu_id=gpu_id)
    
    if is_main_process():
        logger.info('Init: ' + ('On' if cfg.perform_init else 'Off'))
        
    if cfg.test:
        if args.weights:
            target_models = [args.weights]
            logger.info(f"Testing specified model: {args.weights}")
        else:
            target_models = []
        test(trainer, dir, checkpoint_paths=target_models)
        if is_distributed: dist.barrier()
    else:
        best_models = train(trainer, dir)
        if is_distributed: dist.barrier()
        test(trainer, dir, checkpoint_paths=best_models)
        if is_distributed: dist.barrier()
        
    if is_distributed:
        dist.destroy_process_group()

class customAugment(RandAugment):
    def __init__(self, num_ops: int = 2, magnitude: int = 8, num_magnitude_bins: int = 31, interpolation: InterpolationMode = InterpolationMode.NEAREST, fill: Tuple[float] = None) -> None:
        super().__init__()
    # [修复] 修正类型提示中的 Dict 错误
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> dict: 
        return {
            "Identity": (torch.tensor(0.0), False), "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True), "TranslateX": (torch.linspace(0.0, 150.0/331.0*image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0/331.0*image_size[0], num_bins), True), "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True), "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True), "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "AutoContrast": (torch.tensor(0.0), False), "Equalize": (torch.tensor(0.0), False),
        }

if __name__ == '__main__':
    main()