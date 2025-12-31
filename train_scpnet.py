import math
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
    iWAN, G_AN, LL, Weighted_Hill, Modified_VLPL,GPRLoss,BBAMLossVisual, GCELoss,SCELoss,Hill_Consistency,SPLC_Consistency,Hill_Ignore
)
from utils import AverageMeter, add_weight_decay, mAP, compute_recall_at_k
from config import cfg
from consistency import ConsistencyAugmentor

from model import (
    load_clip_model, MMLSurgAdapt, Resnet, ViT, CrossModel, CLIP_for_train, VLPL, HSPNet,
    
    MMLSurgAdaptSCPNet
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

        # [SCPNet]
        elif model_name == 'SCPNet': self.model = MMLSurgAdaptSCPNet(classnames, clip_model)
        elif model_name == 'SCPNet_Plus': 
            print("-" * 50)
            print(f"[DEBUG CHECK] Reading Config:")
            print(f" >> Alpha (SGLC): {getattr(cfg, 'sglc_alpha', 'Not Found')}")
            print(f" >> Threshold (SPP): {getattr(cfg, 'sim_threshold', 'Not Found')}")
            print(f" >> Top-K (SPP): {getattr(cfg, 'top_k', 'Not Found')}")
            print("-" * 50)
            # ==================================================

            self.model = MMLSurgAdaptSCPNet_Plus(
                classnames, 
                clip_model,
                alpha=getattr(cfg, 'sglc_alpha', 0.1),         
                sim_threshold=getattr(cfg, 'sim_threshold', 0.25), 
                top_k=getattr(cfg, 'top_k', 10),
                external_adj_path=getattr(cfg, 'external_adj_path', None)
                               
            )
        else:
            raise NameError(f"Model '{model_name}' not recognized.")

        logger.info(f"Successfully initialized model: {model_name}")
        self.classnames = classnames

        if self.distributed:
            # 将模型中所有的 nn.BatchNorm 转换为 nn.SyncBatchNorm（保证跨卡统计一致）
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logger.info("已开启同步 BatchNorm (SyncBatchNorm)")

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
        debug = getattr(cfg, "debug", False)
        debug_step = getattr(cfg, "debug_step", 200)

        image = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # 特殊损失：需要 image encoder 特征（例如 BBAM 等）
        if getattr(criterion, 'needs_features', False):
            with autocast():
                features = self.model_unwrap.image_encoder(image.type(self.model_unwrap.dtype))
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            loss, _ = criterion(features, target, epoch)
            return loss

        # 常规：直接 forward(image) + loss
        with autocast():
            # SCPNet：可能返回 (logits, ignore_neg_mask) 或 logits
            if cfg.model in ['SCPNet', 'SCPNet_Plus', 'MMLSurgAdaptSCPNet_Plus']:
                out = self.model(image)

                ignore_neg_mask = None
                if isinstance(out, tuple):
                    if len(out) >= 2:
                        logits, ignore_neg_mask = out[0], out[1]
                    else:
                        logits = out[0]
                else:
                    logits = out

                # Hill_Ignore 等：如果支持 ignore_neg_mask 就传
                try:
                    loss, _ = criterion(logits, target, epoch, ignore_neg_mask=ignore_neg_mask)
                except TypeError:
                    loss, _ = criterion(logits, target, epoch)

            # 其它模型：兼容 tuple 返回
            else:
                out = self.model(image).float()
                if isinstance(out, tuple):
                    out = out[0]
                loss, _ = criterion(out, target, epoch)

        if debug and is_main_process() and epoch_i % debug_step == 0:
            with torch.no_grad():
                probs = torch.sigmoid(logits)

                # 正样本数（GT）
                pos_cnt = target.sum(dim=1).float().mean().item()

                # 预测 top1 置信度
                top1_prob = probs.max(dim=1)[0].mean().item()

                # ignore 掩码情况（若存在）
                if ignore_neg_mask is not None:
                    ignore_cnt = ignore_neg_mask.sum(dim=1).float().mean().item()
                else:
                    ignore_cnt = 0.0

                logger.info(
                    f"[DEBUG][E{epoch}][I{epoch_i}] "
                    f"loss={loss.item():.4f} | "
                    f"pos/gt={pos_cnt:.2f} | "
                    f"top1_prob={top1_prob:.3f} | "
                    f"ignore/cls={ignore_cnt:.2f}"
                )
        if debug and is_main_process() and epoch_i % debug_step == 0:
            with torch.no_grad():
                pred_idx = logits.argmax(dim=-1)
                b0 = 0  # 只看 batch 第一个样本

                logger.info(
                    f"[DEBUG][SCP] "
                    f"pred_idx={pred_idx[b0].item()} | "
                    f"pred_logit={logits[b0, pred_idx[b0]].item():.3f}"
                )

                if ignore_neg_mask is not None:
                    ignored = torch.where(ignore_neg_mask[b0])[0].tolist()
                    logger.info(f"[DEBUG][SCP] ignored_labels(sample0)={ignored}")


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
        'Hill_Ignore': Hill_Ignore
    }
    if criterion is None:
        criterion = loss_dict.get(cfg.loss, lambda: None)()
        if torch.cuda.is_available() and criterion is not None: criterion = criterion.cuda()
    
    sigmoid = torch.sigmoid
    preds_regular, preds_ema, all_vids, targets = [], [], [], []
    recall_logits, recall_targets = [], []
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
        recall_logits.append(output_ema_logits.detach().cpu())
        recall_targets.append(target.detach().cpu())

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

    recall_at_k_results = compute_recall_at_k(
        torch.cat(recall_logits, dim=0),
        torch.cat(recall_targets, dim=0),
        ks=(1, 3, 5, 10),
        # TODO: pass class_mask when View/Triplet mappings are available
    )
    logger.info("Recall@K (ALL classes):")
    logger.info("K\tRecall@K\tHit\tTotal_Pos")
    for k in sorted(recall_at_k_results.keys()):
        res = recall_at_k_results[k]
        logger.info(f"{k}\t{res['recall']:.4f}\t{res['hit']}\t{res['total_pos']}")

    os.makedirs("outputs", exist_ok=True)
    recall_json_path = os.path.join("outputs", f"recall_at_k_val_epoch{epoch}.json")
    with open(recall_json_path, "w") as f:
        json.dump(recall_at_k_results, f, indent=2)
    logger.info(f"Recall@K results saved to {recall_json_path}")
    
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
        'Hill_Ignore': Hill_Ignore
    }
    criterion = loss_dict.get(cfg.loss, lambda: None)()
    if criterion is None: raise ValueError(f"Loss function '{cfg.loss}' not found.")
    if torch.cuda.is_available(): criterion = criterion.cuda()
    if is_main_process(): print(f"Using criterion: {criterion}")
    
    parameters = add_weight_decay(trainer.model, cfg.weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=cfg.lr, weight_decay=0)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    steps_per_epoch = len(trainer.train_loader)
    accumulation_steps = getattr(cfg, 'accumulation_steps', 1)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / accumulation_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * cfg.epochs

    mimic_single_card = getattr(cfg, "mimic_single_card", False)

    # 支持 per_step / per_epoch 两种调度粒度；若开启 mimic_single_card，默认 per_step 以对齐单卡步数
    scheduler_mode = getattr(
        cfg,
        "scheduler_granularity",
        "per_step" if mimic_single_card else "per_epoch",
    )
    if scheduler_mode not in ["per_step", "per_epoch"]:
        scheduler_mode = "per_epoch"

    scheduler_T_max = total_optimizer_steps if scheduler_mode == "per_step" else cfg.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_T_max, eta_min=1e-6
    )
    scaler = GradScaler()

    if is_main_process():
        effective_batch = cfg.batch_size * world_size * accumulation_steps
        logger.info(
            f"Effective batch size (global): {effective_batch} = batch_per_gpu({cfg.batch_size})"
            f" x world_size({world_size}) x accumulation({accumulation_steps})"
        )

    # 保存每个 epoch 的 checkpoint，不再裁剪
    saved_paths = []
    save_dir_base = os.path.join(cfg.checkpoint, dir)
    if is_main_process(): os.makedirs(save_dir_base, exist_ok=True)

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
                if scheduler_mode == "per_step":
                    scheduler.step()
            
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
            
            torch.save(state_dict, filepath)
            saved_paths.append(filepath)
            logger.info(f"Saved model for epoch {epoch}: {filename}")

        trainer.model.train()
        if scheduler_mode == "per_epoch":
            scheduler.step()

    final_paths = saved_paths if is_main_process() else []
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
        recall_logits, recall_targets = [], []
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
            recall_logits.append(output_logits.detach().cpu())
            recall_targets.append(target.detach().cpu())

        all_labels = np.concatenate(targets, axis=0)
        all_predictions = np.concatenate(preds, axis=0)
        all_vids_np = np.array(all_vids)
        method_name = filename.replace('.ckpt', '')
        calculate_metrics(all_labels, all_predictions, all_predictions, all_vids_np, True, dir, method=method_name)
        recall_at_k_results = compute_recall_at_k(
            torch.cat(recall_logits, dim=0),
            torch.cat(recall_targets, dim=0),
            ks=(1, 3, 5, 10),
            # TODO: pass class_mask when View/Triplet mappings are available
        )
        logger.info(f"Recall@K (ALL classes) for {method_name}:")
        logger.info("K\tRecall@K\tHit\tTotal_Pos")
        for k in sorted(recall_at_k_results.keys()):
            res = recall_at_k_results[k]
            logger.info(f"{k}\t{res['recall']:.4f}\t{res['hit']}\t{res['total_pos']}")

        os.makedirs("outputs", exist_ok=True)
        recall_json_path = os.path.join("outputs", f"recall_at_k_test_{method_name}.json")
        with open(recall_json_path, "w") as f:
            json.dump(recall_at_k_results, f, indent=2)
        logger.info(f"Recall@K results saved to {recall_json_path}")
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
    
    # 可配置的确定性与 TF32 行为；mimic_single_card 时默认开启确定性并关闭 TF32
    mimic_single_card = getattr(cfg, "mimic_single_card", False)
    strict_deterministic = getattr(cfg, "strict_deterministic", mimic_single_card)
    allow_tf32 = getattr(cfg, "allow_tf32", not mimic_single_card)

    if strict_deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    if is_main_process():
        logger.info(
            f"Deterministic={strict_deterministic}, TF32={allow_tf32}, "
            f"Scheduler mode={getattr(cfg, 'scheduler_granularity', 'per_epoch')}"
        )
    
    # 若希望多卡模拟单卡全局 batch，按 target_global_batch 均分到每卡
    target_global_batch = getattr(cfg, "target_global_batch", None)
    world_size = dist.get_world_size() if is_distributed else 1
    if target_global_batch is not None and world_size > 1:
        per_gpu_batch = max(1, math.ceil(target_global_batch / world_size))
        if is_main_process():
            logger.info(
                f"Use target_global_batch={target_global_batch}, "
                f"world_size={world_size}, set batch_size per GPU -> {per_gpu_batch}"
            )
        cfg.batch_size = per_gpu_batch

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
