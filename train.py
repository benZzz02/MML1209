import torch  # <--- 必须放在第一行，最先导入！
import torch.distributed as dist
import os
import time
import warnings
import argparse
import random
import json
from datetime import datetime
from statistics import mean
from typing import Tuple
import torch.distributed as dist
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score, average_precision_score
from datetime import timedelta
# 项目内部模块导入
import ivtmetrics
import args
from log import logger
from loss import (
    SPLC, GRLoss, Hill, AsymmetricLossOptimized, WAN, VLPL_Loss, 
    iWAN, G_AN, LL, Weighted_Hill, Modified_VLPL,GPRLoss,BBAMLossVisual, GCELoss,SCELoss,Hill_Consistency,SPLC_Consistency
)
from mmlsurgadapt import MMLSurgAdaptTrainer
from utils import AverageMeter, add_weight_decay, mAP, estimate_class_distribution, run_cap_procedure,TopKCheckpointManager
from config import cfg  # isort:skip

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
# 指标计算与处理函数 (仅主进程运行)
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

# [train.py] 替换原有的 calculate_metrics
def calculate_metrics(labels, preds, preds_ema, video_ids, test, dir, method=None):
    if not is_main_process(): return None, None, None # 修改返回值

    labels = np.round(labels)
    datasets = ["cholec80", "endoscapes", "cholect50"]
    a, b, c = None, None, None

    for dataset in datasets:
        # print(f"Processing dataset: {dataset}") 
        mask = np.array([dataset in v for v in video_ids])

        if not np.any(mask):
            # print(f"No samples for dataset: {dataset}, skipping.")
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
        # print("Warning: Not all dataset metrics were calculated.")
    
    print("Calculating metrics done")
    return a, b, c  # [新增] 返回计算结果

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
# 验证与测试流程 (核心修正：主卡验证)
# =============================================================================
def validate(trainer, epoch: int, dir, criterion=None) -> dict:
    # 非主进程直接进入等待状态
    if not is_main_process():
        if dist.is_initialized():
            dist.barrier()
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
        'Hill_Consistency': Hill_Consistency,
        'SPLC_Consistency': SPLC_Consistency,
    }
    criterion = loss_dict.get(cfg.loss, lambda: None)()
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    
    sigmoid = torch.sigmoid
    preds_regular, preds_ema, all_vids, targets = [], [], [], []
    losses, losses_ema = [], []

    # 使用 .module 获取原始模型，避免 DDP sync 开销
    model_to_run = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    ema_to_run = trainer.ema.module if hasattr(trainer.ema, 'module') else trainer.ema

    # 验证集 DataLoader (DDP修复后为全量数据)
    for _, (input, target, vid) in enumerate(trainer.val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        with torch.no_grad():
            with autocast():
                output_logits = model_to_run(input)
                output_ema_logits = ema_to_run(input)
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
                    output_logits = model_to_run(input)
                    output_ema_logits = ema_to_run(input)

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

    # 验证结束，主进程解除屏障
    if dist.is_initialized():
        dist.barrier()

    return evals

def init_validate(trainer, epoch: int):
    if not is_main_process():
        if dist.is_initialized():
            dist.barrier()
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
                output_logits = model_to_run(input)
                output_ema_logits = ema_to_run(input)

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
    
    if dist.is_initialized():
        dist.barrier()

    return loss_min, if_ema_better

def save_best_init(trainer, if_ema_better, dir):
    if not is_main_process(): return
    # Logic same as save_best but for init phase
    state_dict = trainer.model.module.state_dict() if hasattr(trainer.model, "module") else trainer.model.state_dict()
    ema_state_dict = trainer.ema.module.state_dict() if hasattr(trainer.ema, "module") else trainer.ema.state_dict()
    
    save_path = os.path.join(cfg.checkpoint, f'{dir}/init')
    os.makedirs(save_path, exist_ok=True)
    
    if if_ema_better:
        torch.save(ema_state_dict, os.path.join(save_path, 'model-highest.ckpt'))
    else:
        torch.save(state_dict, os.path.join(save_path, 'model-highest.ckpt'))

# [train.py] 修正后的 train 函数
def train(trainer, dir) -> list:
    # --- 原始设置保持不变 ---
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
        'Hill_Consistency': Hill_Consistency,
        'SPLC_Consistency': SPLC_Consistency,
    }
    criterion = loss_dict.get(cfg.loss, lambda: None)()
    if criterion is None:
        raise ValueError(f"Loss function '{cfg.loss}' not found.")
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    if is_main_process():
        print(f"Using criterion: {criterion}")
    
    parameters = add_weight_decay(trainer.model, cfg.weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=cfg.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # [恢复] 获取步数用于日志
    steps_per_epoch = len(trainer.train_loader)
    accumulation_steps = getattr(cfg, 'accumulation_steps', 1)

    # --- [Top-K 简单列表] ---
    top_k_num = getattr(cfg, 'top_k', 3)
    best_map_list = []      # 存 (score, epoch, filepath)
    best_pp_loss_list = []  # 存 (score, epoch, filepath)
    best_sp_loss_list = []  # 存 (score, epoch, filepath)
    
    # 基础保存路径
    save_dir_base = os.path.join(cfg.checkpoint, dir)
    if is_main_process():
        os.makedirs(save_dir_base, exist_ok=True)
    # -----------------------

    # CAP 逻辑
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

    # --- Initialization Phase (完全恢复您的代码) ---
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
                    output = trainer.model(image).float()
                loss = nn.BCEWithLogitsLoss()(output, target)
                scaler.scale(loss).backward()
                scaler.step(init_optimizer)
                scaler.update()
                trainer.ema.update(trainer.model)

                # [恢复] 这里的日志格式
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

    # --- Main Training Phase ---
    if is_main_process(): logger.info(f"Gradient Accumulation Steps: {accumulation_steps}")

    for epoch in range(cfg.epochs):
        if hasattr(trainer.train_loader, 'sampler') and hasattr(trainer.train_loader.sampler, 'set_epoch'):
             trainer.train_loader.sampler.set_epoch(epoch)

        # CAP Procedure
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
            
            # [恢复] 严格保留您的日志格式 (含 LR 和 log_loss 计算)
            if i % 100 == 0 and is_main_process():
                log_loss = loss.item() * accumulation_steps
                logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.4f}'
                    .format(epoch, cfg.epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3), cfg.lr, log_loss))

        # Validation
        evals = validate(trainer, epoch, dir, criterion=criterion)

        # ================= [Top-K 简单处理逻辑] =================
        if is_main_process() and evals:
            cur_map = evals['pp_map']
            cur_pp_loss = evals['pp_loss']
            
            # 决定使用 EMA 还是 Regular
            use_ema = evals.get('pp_map_if_better', False)
            if use_ema and hasattr(trainer, 'ema'):
                state_dict = trainer.ema.module.state_dict() if hasattr(trainer.ema, "module") else trainer.ema.state_dict()
                suffix = "ema"
            else:
                state_dict = trainer.model.module.state_dict() if hasattr(trainer.model, "module") else trainer.model.state_dict()
                suffix = "reg"
            
            # 构造文件名
            filename = f"epoch_{epoch}_mAP_{cur_map:.2f}_loss_{cur_pp_loss:.4f}_{suffix}.ckpt"
            filepath = os.path.join(save_dir_base, filename)
            
            save_needed = False
            
            # 1. mAP (Descending)
            best_map_list.append((cur_map, epoch, filepath))
            best_map_list.sort(key=lambda x: x[0], reverse=True)
            # 如果当前 epoch 在前 K 名，就保存
            if any(x[1] == epoch for x in best_map_list[:top_k_num]):
                save_needed = True

            # 2. PP Loss (Ascending)
            best_pp_loss_list.append((cur_pp_loss, epoch, filepath))
            best_pp_loss_list.sort(key=lambda x: x[0])
            if any(x[1] == epoch for x in best_pp_loss_list[:top_k_num]):
                save_needed = True
                
            # 3. SP Loss (Ascending, optional)
            if cfg.val_sp and 'sp_loss' in evals:
                best_sp_loss_list.append((evals['sp_loss'], epoch, filepath))
                best_sp_loss_list.sort(key=lambda x: x[0])
                if any(x[1] == epoch for x in best_sp_loss_list[:top_k_num]):
                    save_needed = True
            
            # 保存文件
            if save_needed:
                torch.save(state_dict, filepath)
                logger.info(f"Saved Top-K model: {filename}")

            # [清理垃圾]
            # 计算白名单（所有榜单的前 K 名对应的路径）
            valid_paths = set()
            valid_paths.update([x[2] for x in best_map_list[:top_k_num]])
            valid_paths.update([x[2] for x in best_pp_loss_list[:top_k_num]])
            if cfg.val_sp:
                valid_paths.update([x[2] for x in best_sp_loss_list[:top_k_num]])
            
            # 扫描目录并删除不在白名单里的 .ckpt
            for f in os.listdir(save_dir_base):
                if f.endswith('.ckpt') and "epoch_" in f:
                    full_p = os.path.join(save_dir_base, f)
                    if full_p not in valid_paths:
                        try:
                            os.remove(full_p)
                            logger.info(f"Removed worse model: {f}")
                        except: pass
            
            # 修剪列表内存
            best_map_list = best_map_list[:top_k_num]
            best_pp_loss_list = best_pp_loss_list[:top_k_num]
            if cfg.val_sp:
                best_sp_loss_list = best_sp_loss_list[:top_k_num]

        trainer.model.train()
        scheduler.step()

    # 返回给 test
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
    
    # 如果没传路径，兼容一下（比如纯测试模式）
    target_paths = checkpoint_paths if checkpoint_paths else []
    
    if not target_paths and cfg.test:
        # 只有在纯测试模式下才允许扫描
        import glob
        base = os.path.join(cfg.checkpoint, dir)
        target_paths = glob.glob(os.path.join(base, "*.ckpt"))
        
    if not target_paths:
        logger.warning("No checkpoints provided to test.")
        return

    # 按文件名排序
    target_paths.sort()
    
    sigmoid = torch.sigmoid
    processed_epochs = set() # [去重关键]

    for ckpt_path in target_paths:
        if not os.path.exists(ckpt_path): continue
        
        filename = os.path.basename(ckpt_path)
        
        # 提取 epoch 进行去重
        # 假设文件名格式: epoch_12_...
        epoch_id = "unknown"
        try:
            parts = filename.split('_')
            if parts[0] == 'epoch':
                epoch_id = parts[1]
        except: pass
        
        if epoch_id != "unknown" and epoch_id in processed_epochs:
            logger.info(f"Skipping {filename} (Epoch {epoch_id} already tested)")
            continue
            
        logger.info(f"Testing checkpoint: {filename} ...")
        
        # 加载权重
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
                output = sigmoid(model_to_run(input))
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            all_vids.extend(vid)

        all_labels = np.concatenate(targets, axis=0)
        all_predictions = np.concatenate(preds, axis=0)
        all_vids_np = np.array(all_vids)
        
        # 计算并保存结果 (method名用去掉后缀的文件名)
        method_name = filename.replace('.ckpt', '')
        calculate_metrics(all_labels, all_predictions, all_predictions, all_vids_np, True, dir, method=method_name)
        
        # 标记为已测试
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

    trainer = MMLSurgAdaptTrainer(distributed=is_distributed, gpu_id=gpu_id)
    
    if is_main_process():
        logger.info('Init: ' + ('On' if cfg.perform_init else 'Off'))
        
    if cfg.test:
        if args.weights:
            # 如果传入了 --weights，就只测这一个
            target_models = [args.weights]
            logger.info(f"Testing specified model: {args.weights}")
        else:
            # 没传就传空列表，让 test 函数自己去扫描文件夹
            target_models = []
        test(trainer, dir, checkpoint_paths=target_models)
        if is_distributed: dist.barrier()
    else:
        # 1. 训练，并获取最佳模型列表
        best_models = train(trainer, dir)
        
        # 2. 同步
        if is_distributed: dist.barrier()
            
        # 3. 传入列表进行测试
        test(trainer, dir, checkpoint_paths=best_models)
        
        if is_distributed: dist.barrier()
        
    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()