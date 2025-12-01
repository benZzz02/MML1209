import os
import random
from copy import deepcopy

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torchvision import datasets as datasets

from config import cfg
from log import logger

import torch.distributed as dist
from torch.cuda.amp import autocast

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0  # type: ignore
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):

    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01  # type: ignore


class CocoDetection(datasets.coco.CocoDetection):

    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)

    def labels(self):
        return [v["name"] for v in self.coco.cats.values()]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:  # type: ignore
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output

        path = coco.loadImgs(img_id)[0]['file_name']  # type: ignore
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target = target.max(dim=0)[0]
        return img, target


class COCO_missing_dataset(torch.utils.data.Dataset):  # type: ignore

    def __init__(self,
                 root,
                 annFile,
                 transform=None,
                 target_transform=None,
                 class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        # name = names.strip('\n').split(' ')
        self.name = names
        # self.label = name[:,1]
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform

    def __getitem__(self, index):
        name = self.name[index]
        path = name.strip('\n').split(',')[0]
        num = name.strip('\n').split(',')[1]
        num = num.strip(' ').split(' ')
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        if os.path.exists(os.path.join(self.root, path)) == False:
            label = np.zeros([self.class_num])
            label = torch.tensor(label, dtype=torch.long)
            img = np.zeros((448, 448, 3))
            img = Image.fromarray(np.uint8(img))  # type: ignore
            exit(1)
        else:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)  # type: ignore # noqa
        assert (self.target_transform is None)
        return [index,img], label

    def __len__(self):
        return len(self.name)

    def labels(self):
        if "coco" in cfg.data:
            assert (False)
        elif "nuswide" in cfg.data:
            with open('nuswide_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "voc" in cfg.data:
            with open('voc_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "cub" in cfg.data:
            with open('cub_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        else:
            assert (False)


class COCO_missing_val_dataset(torch.utils.data.Dataset):  # type: ignore

    def __init__(self,
                 root,
                 annFile,
                 transform=None,
                 target_transform=None,
                 class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        # name = names.strip('\n').split(' ')
        self.name = names
        # self.label = name[:,1]
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform

    def __getitem__(self, index):
        name = self.name[index]
        path = name.strip('\n').split(',')[0]
        num = name.strip('\n').split(',')[1]
        num = num.strip(' ').split(' ')
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        if os.path.exists(os.path.join(self.root, path)) == False:
            label = np.zeros([self.class_num])
            label = torch.tensor(label, dtype=torch.long)
            img = np.zeros((448, 448, 3))
            img = Image.fromarray(np.uint8(img))  # type: ignore
            exit(1)
        else:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)  # type: ignore # noqa
        assert (self.target_transform is None)
        return img, label

    def __len__(self):
        return len(self.name)

    def labels(self):
        if "coco" in cfg.data:
            assert (False)
        elif "nuswide" in cfg.data:
            with open('nuswide_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "voc" in cfg.data:
            with open('voc_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        elif "cub" in cfg.data:
            with open('cub_labels.txt', 'r') as f:
                text = f.read()
            return text.split('\n')
        else:
            assert (False)


class ModelEma(torch.nn.Module):

    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(),
                                      model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model,
                     update_fn=lambda e, m: self.decay * e +
                     (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class CutoutPIL(object):

    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255),
                      random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)  # type: ignore

        return x


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    gcn = []
    gcn_no_decay = []
    prefix = "module." if torch.cuda.device_count() > 1 else "" 
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if name.startswith(f"{prefix}gc"):
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                gcn_no_decay.append(param)
            else:
                gcn.append(param)
            assert("gcn" in cfg.model_name)
        elif len(param.shape) == 1 or name.endswith(
                ".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{
        'params': no_decay,
        'weight_decay': 0.
    }, {
        'params': decay,
        'weight_decay': weight_decay
    }, {
        'params': gcn_no_decay,
        'weight_decay': 0.
    }, {
        'params': gcn,
        'weight_decay': weight_decay
    }]

def get_ema_co():
    if "coco" in cfg.data:
        ema_co = np.exp(np.log(0.82)/(641*cfg.ratio))  # type: ignore
        # ema_co = 0.9997
    elif "nus" in cfg.data:
        ema_co = np.exp(np.log(0.82)/(931*cfg.ratio))  # type: ignore
        # ema_co = 0.9998
    elif "voc" in cfg.data:
        ema_co = np.exp(np.log(0.82)/(45*cfg.ratio))  # type: ignore
        # ema_co = 0.9956
    elif "cub" in cfg.data:
        if cfg.batch_size == 96:
            ema_co = np.exp(np.log(0.82)/(63*cfg.ratio))
        else:
            ema_co = np.exp(np.log(0.82)/(47*cfg.ratio))  # type: ignore
    elif "cholec" in cfg.data:
        ema_co = np.exp(np.log(0.82)/(641*cfg.ratio))
    else:
        assert(False)
    return ema_co

# =============================================================================
# [CAP] Class-Aware Pseudo-Labeling Utils
# =============================================================================

def estimate_class_distribution(dataset, num_classes):
    """
    [CAP Step 1] 计算已标注数据的类别分布 (Pos Frequency)
    对应论文中的 gamma_hat
    """
    logger.info("[CAP] Estimating class distribution from dataset...")
    
    # 1. 尝试直接获取 labels (CholecDataset / Cholec_Train)
    targets = None
    
    # 情况 A: Cholec_Train (self.data 是 list)
    if hasattr(dataset, 'data') and isinstance(dataset.data, list):
        try:
            # self.data 结构: [[path, label_tensor, vid], ...]
            # 我们只提取 label_tensor
            all_labels = [d[1].numpy() if isinstance(d[1], torch.Tensor) else d[1] for d in dataset.data]
            targets = np.array(all_labels)
        except Exception as e:
            logger.warning(f"[CAP] Failed to parse 'dataset.data' list: {e}")

    # 情况 B: 拥有 .labels 属性 (如果未来更改了实现)
    if targets is None and hasattr(dataset, 'labels') and not callable(dataset.labels):
         targets = np.array(dataset.labels)

    if targets is None:
        logger.error("[CAP] Could not find labels in dataset to estimate distribution.")
        return None

    n_total = len(targets)
    if n_total == 0: return None

    # 计算每个类别的正样本频率
    # axis=0 表示沿样本维度求和
    pos_freq = np.sum(targets == 1, axis=0) / n_total
    
    logger.info(f"[CAP] Distribution estimated. Max freq: {pos_freq.max():.4f}, Min freq: {pos_freq.min():.4f}")
    return pos_freq
# 文件: utils.py (覆盖之前的 run_cap_procedure)

@torch.no_grad()
def run_cap_procedure(trainer, loader, pos_freq, device, ratio=1.0):
    """
    [CAP Step 2 - Global Ranking Version]
    全局预测 -> 全局汇总 -> 全局排序定阈值 -> 本地更新
    """
    if pos_freq is None: return

    logger.info(f">>> [CAP] Running CAP (Global Ranking): Ratio={ratio}...")
    model = trainer.model
    model.eval()
    
    # 1. 本地推理 (Local Inference)
    local_preds_list = []
    local_indices_list = []
    
    for _, batch_data in enumerate(loader):
        input = batch_data[0].to(device, non_blocking=True)
        idx = batch_data[-1]
        
        with autocast():
            logits = model(input) # 形状 [B, 110]
            
            # =========== [修改开始] 针对不同任务分段计算概率 ===========
            
            # 任务1: 阶段 (0-6) -> 假设是单标签互斥任务，使用 Softmax
            logits_phase = logits[:, :7]
            probs_phase = torch.softmax(logits_phase, dim=1)
            
            # 任务2: Endoscapes (7-9) -> 视具体定义而定，这里假设保持 Sigmoid
            logits_endo = logits[:, 7:10]
            probs_endo = torch.sigmoid(logits_endo)
            
            # 任务3: 细粒度动作 (10-110) -> 多标签任务，使用 Sigmoid
            logits_action = logits[:, 10:]
            probs_action = torch.sigmoid(logits_action)
            
            # 将三段概率拼接回 [B, 110] 的大矩阵
            probs = torch.cat([probs_phase, probs_endo, probs_action], dim=1)
            
            # =========== [修改结束] ===================================
        
        local_preds_list.append(probs)
        local_indices_list.append(idx.to(device))
    if len(local_preds_list) > 0:
            local_probs = torch.cat(local_preds_list, dim=0)      # [N_local, 110]
            local_indices = torch.cat(local_indices_list, dim=0)  # [N_local]
    else:
            # 防止有些卡分不到数据的情况（虽然很少见）
        return
    # 2. 全局汇总 (Global Gather) - 仅在 DDP 模式下触发
    if dist.is_initialized():
        world_size = dist.get_world_size()
        
        # 准备容器接收所有卡的结果
        # 注意：DDP 要求各卡 Tensor 尺寸一致。DistributedSampler 默认会补齐数据(padding)，所以通常是对齐的。
        gathered_probs = [torch.zeros_like(local_probs) for _ in range(world_size)]
        
        # 全局收集预测概率
        dist.all_gather(gathered_probs, local_probs)
        
        # 拼接成全局大矩阵 [N_total, C]
        global_probs = torch.cat(gathered_probs, dim=0)
        
        # (可选) 如果不仅为了计算阈值，还需要全局去重，可以同时 gather indices
        # 但为了简化逻辑，我们只用 global_probs 算阈值，更新还是在本地做
        
    else:
        # 单卡模式，全局就是本地
        global_probs = local_probs

    # 3. 计算全局动态阈值 (Calculate Global Thresholds)
    # 转为 numpy 处理排序 (CPU)
    global_probs_np = global_probs.float().cpu().numpy()
    num_global_samples = global_probs_np.shape[0]
    num_classes = global_probs_np.shape[1]
    
    # 全局排序 (Column-wise)
    # -x 排序等同于 x 降序
    sorted_global = -np.sort(-global_probs_np, axis=0)
    
    # 计算截断位置 (Global Cutoff)
    # Formula: 总样本数 * 真实分布频率 * 缩放系数
    cutoff_indices = np.floor(pos_freq * num_global_samples * ratio).astype(int)
    cutoff_indices = np.clip(cutoff_indices - 1, 0, num_global_samples - 1)
    
    # 得到每个类别的阈值向量 [C]
    thresholds = sorted_global[cutoff_indices, range(num_classes)]
    
    # 打印一下阈值信息，看看有没有离谱的
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(f"[CAP] Global Thresholds - Max: {thresholds.max():.4f}, Min: {thresholds.min():.4f}, Mean: {thresholds.mean():.4f}")

    # 4. 本地生成伪标签 (Local Assignment)
    # 使用全局统一的 thresholds 对 本地数据 进行判定
    # 必须转回 numpy 进行比较 (因为 thresholds 是 numpy)
    local_probs_np = local_probs.float().cpu().numpy()
    
    # 生成伪标签 (大于全局阈值即为 1)
    pseudo_labels = (local_probs_np >= thresholds).astype(np.float32)
    
    # 5. 本地更新 Dataset
    # 只需要更新自己手里的这一部分数据，index 是绝对对齐的
    local_indices_np = local_indices.cpu().numpy()
    
    if hasattr(loader.dataset, 'update_labels'):
        loader.dataset.update_labels(pseudo_labels, local_indices_np)
    else:
        logger.warning("[CAP] Dataset missing 'update_labels'. Skipped.")

    model.train()
    # 同步结束
    if dist.is_initialized(): dist.barrier()