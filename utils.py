import os
import random
from copy import deepcopy
import heapq
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
    [CAP Step 1] 计算每个类别在正样本图片中的占比
    
    阈值[c] = 该类正样本图片数 / 所有正样本图片总数
    """
    logger.info("[CAP] Estimating class distribution from dataset...")
    
    targets = None
    
    if hasattr(dataset, 'data') and isinstance(dataset.data, list):
        try:
            all_labels = [d[1]. numpy() if isinstance(d[1], torch.Tensor) else np.array(d[1]) for d in dataset.data]
            targets = np. array(all_labels)
        except Exception as e:
            logger.warning(f"[CAP] Failed to parse 'dataset.data' list: {e}")

    if targets is None:
        logger.error("[CAP] Could not find labels in dataset.")
        return None

    # targets 形状: [N, 110]，值为 0 或 1
    
    # 统计每个类别的正样本图片数
    pos_count_per_class = np.sum(targets == 1, axis=0)  # 形状 [110]
    
    # 统计所有正样本图片总数（至少有一个类别为1的图片）
    has_positive = np.any(targets == 1, axis=1)  # 形状 [N]，每张图片是否有正样本
    total_positive_images = np.sum(has_positive)
    
    if total_positive_images == 0:
        logger.error("[CAP] No positive samples found.")
        return None
    
    # 每个类别的阈值 = 该类正样本数 / 正样本图片总数
    class_ratio = pos_count_per_class / total_positive_images
    
    logger.info(f"[CAP] Total positive images: {total_positive_images}")
    logger.info(f"[CAP] Class ratio - Max: {class_ratio.max():.4f}, Min: {class_ratio.min():.4f}, Sum: {class_ratio.sum():.4f}")
    
    return class_ratio


@torch.no_grad()
def run_cap_procedure(trainer, loader, class_ratio, device, ratio=1.0):
    """
    [CAP Step 2] 在负样本中生成伪标签
    
    对于每个类别 c:
        1. 找出该类的负样本（target[c] == 0）
        2. 计算预测概率并排序
        3. 选择前 K = 负样本数 × class_ratio[c] × ratio 个作为伪正标签
    """
    if class_ratio is None:
        return

    logger.info(f">>> [CAP] Running CAP: ratio={ratio}...")
    
    if dist.is_initialized():
        num_classes = len(class_ratio) if class_ratio is not None else 110
        
        # 创建 tensor 用于广播
        if dist.get_rank() == 0:
            class_ratio_tensor = torch.from_numpy(np.array(class_ratio)).float().to(device)
        else:
            class_ratio_tensor = torch.zeros(num_classes, dtype=torch.float32, device=device)
        
        # 从 rank 0 广播到所有 rank
        dist.broadcast(class_ratio_tensor, src=0)
        
        # 转回 numpy
        class_ratio = class_ratio_tensor.cpu().numpy()
    
    model = trainer.model
    model.eval()
    
    # 1. 收集所有样本的预测概率和原始标签
    local_preds_list = []
    local_indices_list = []
    local_targets_list = []
    
    for _, batch_data in enumerate(loader):
        input = batch_data[0].to(device, non_blocking=True)
        target = batch_data[1]. to(device, non_blocking=True)
        idx = batch_data[-1]
        
        with autocast():
            logits = model(input)
            
            # Phase (0-6): Softmax
            probs_phase = torch.softmax(logits[:, :7], dim=1)
            # Endoscapes (7-9): Sigmoid
            probs_endo = torch.sigmoid(logits[:, 7:10])
            # Actions (10-110): Sigmoid
            probs_action = torch.sigmoid(logits[:, 10:])
            
            probs = torch.cat([probs_phase, probs_endo, probs_action], dim=1)
        
        local_preds_list.append(probs)
        local_indices_list.append(idx. to(device))
        local_targets_list.append(target)
    
    if len(local_preds_list) == 0:
        return
    
    local_probs = torch.cat(local_preds_list, dim=0)
    local_indices = torch. cat(local_indices_list, dim=0)
    local_targets = torch. cat(local_targets_list, dim=0)
    
    # 2.  全局汇总（DDP）
    if dist.is_initialized():
        world_size = dist. get_world_size()
        
        gathered_probs = [torch.zeros_like(local_probs) for _ in range(world_size)]
        gathered_indices = [torch.zeros_like(local_indices) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(local_targets) for _ in range(world_size)]
        
        dist.all_gather(gathered_probs, local_probs)
        dist.all_gather(gathered_indices, local_indices)
        dist. all_gather(gathered_targets, local_targets)
        
        global_probs = torch.cat(gathered_probs, dim=0)
        global_indices = torch.cat(gathered_indices, dim=0)
        global_targets = torch.cat(gathered_targets, dim=0)
    else:
        global_probs = local_probs
        global_indices = local_indices
        global_targets = local_targets
    
    # 3. 转换为 numpy
    global_probs_np = global_probs.float().cpu().numpy()
    global_indices_np = global_indices.cpu().numpy()
    global_targets_np = global_targets. float().cpu().numpy()
    
    num_samples = global_probs_np.shape[0]
    num_classes = global_probs_np. shape[1]
    
    # 4. 为每个类别生成伪标签（只在负样本中选择）
    global_pseudo_labels = np.zeros_like(global_probs_np, dtype=np.float32)
    pseudo_counts = np.zeros(num_classes, dtype=np.int32)
    
    for c in range(num_classes):
        # 找出该类别的负样本（原始标签为 0）
        negative_mask = (global_targets_np[:, c] == 0)
        num_negatives = negative_mask.sum()
        
        if num_negatives == 0:
            continue
        
        # 计算 K = 负样本数 × class_ratio[c] × ratio
        k = int(np.floor(num_negatives * class_ratio[c] * ratio))
        
        if k <= 0:
            continue
        
        # 获取负样本的索引和概率
        negative_indices = np.where(negative_mask)[0]
        negative_probs = global_probs_np[negative_indices, c]
        
        # 按概率降序排序
        sorted_order = np.argsort(-negative_probs)
        
        # 选择前 K 个
        top_k_in_negatives = sorted_order[:k]
        top_k_global_indices = negative_indices[top_k_in_negatives]
        
        # 标记为伪正标签
        global_pseudo_labels[top_k_global_indices, c] = 1.0
        pseudo_counts[c] = k
    
    # 5. 打印统计信息
    if not dist.is_initialized() or dist.get_rank() == 0:
        total_pseudo = pseudo_counts.sum()
        logger.info(f"====== [CAP Stats] Total Pseudo Positives: {total_pseudo} ======")
        
        stats_str_list = [f"C{c}:{count}" for c, count in enumerate(pseudo_counts)]
        chunk_size = 10
        for i in range(0, num_classes, chunk_size):
            chunk = stats_str_list[i:i+chunk_size]
            logger.info(" ".join(chunk))
        logger.info("================================================================")
    
    # 6. 分发伪标签回本地，并更新 Dataset
    index_to_position = {int(idx): pos for pos, idx in enumerate(global_indices_np)}
    
    local_indices_np = local_indices. cpu().numpy()
    local_pseudo_labels = np.zeros((len(local_indices_np), num_classes), dtype=np. float32)
    
    for i, local_idx in enumerate(local_indices_np):
        global_pos = index_to_position. get(int(local_idx), -1)
        if global_pos >= 0:
            local_pseudo_labels[i] = global_pseudo_labels[global_pos]
    
    # 更新 Dataset（覆盖模式）
    if hasattr(loader.dataset, 'update_labels'):
        loader. dataset.update_labels(local_pseudo_labels, local_indices_np)
    else:
        logger.warning("[CAP] Dataset missing 'update_labels'.  Skipped.")
    
    model.train()
    if dist.is_initialized():
        dist. barrier()
        
        
class TopKCheckpointManager(object):
    def __init__(self, k=3, mode='max', save_dir='checkpoints', metric_name='score'):
        """
        Args:
            k (int): 保存前K个模型
            mode (str): 'max' (如 mAP) 或 'min' (如 Loss)
            save_dir (str): 保存路径
            metric_name (str): 指标名称，用于文件名
        """
        self.k = k
        self.mode = mode
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.top_k = []  # 存储堆元素: (sort_score, epoch, filepath)
        self.best_paths = [] # 存储路径方便检索
        
        if dist.is_initialized():
            if dist.get_rank() == 0:
                os.makedirs(save_dir, exist_ok=True)
        else:
            os.makedirs(save_dir, exist_ok=True)

    def update(self, score, epoch, trainer, if_ema_better):
        """
        更新 Top-K 堆，如果当前分数足够好，则保存模型并移除最差的模型。
        """
        # 仅主进程执行保存操作
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        # 统一转换为最小堆处理：如果是求最大值(max)，存负数；如果是求最小值(min)，存正数
        # 这样堆顶永远是"最差"的那个（数值最大的那个被淘汰，或者负数最大的那个被淘汰）
        sort_score = score if self.mode == 'min' else -score
        
        # 准备保存逻辑
        state_dict = trainer.model.module.state_dict() if hasattr(trainer.model, "module") else trainer.model.state_dict()
        if if_ema_better and hasattr(trainer, 'ema'):
            state_dict = trainer.ema.module.state_dict() if hasattr(trainer.ema, "module") else trainer.ema.state_dict()
            
        filename = f"epoch_{epoch}_{self.metric_name}_{score:.4f}.ckpt"
        filepath = os.path.join(self.save_dir, filename)

        # 堆未满，直接加入
        if len(self.top_k) < self.k:
            heapq.heappush(self.top_k, (-sort_score, epoch, filepath)) # 存入堆时取反sort_score，以便堆顶是"最差"的
            self.save_checkpoint(state_dict, filepath)
            self.best_paths.append(filepath)
            logger.info(f"[{self.metric_name}] Top-K list not full. Added epoch {epoch} (Score: {score:.4f})")
            return

        # 堆已满，比较当前分数与堆顶（目前TopK里最差的）
        # 注意：因为我们存的是 -sort_score，堆顶是最小的，也就是 sort_score 最大的，也就是"最差"的
        worst_neg_score, _, worst_filepath = self.top_k[0]
        
        # 检查是否优于最差分数
        # 我们存的是 -sort_score。如果当前 -sort_score > 堆顶，说明当前更好
        if (-sort_score) > worst_neg_score:
            # 弹出最差的
            heapq.heappop(self.top_k)
            # 删除旧文件
            if os.path.exists(worst_filepath):
                os.remove(worst_filepath)
                logger.info(f"[{self.metric_name}] Removed old checkpoint: {worst_filepath}")
            if worst_filepath in self.best_paths:
                self.best_paths.remove(worst_filepath)

            # 推入新的
            heapq.heappush(self.top_k, (-sort_score, epoch, filepath))
            self.save_checkpoint(state_dict, filepath)
            self.best_paths.append(filepath)
            logger.info(f"[{self.metric_name}] Updated Top-K. Added epoch {epoch} (Score: {score:.4f})")
        else:
            pass
            # logger.info(f"[{self.metric_name}] Epoch {epoch} (Score: {score:.4f}) did not enter Top-{self.k}")

    def save_checkpoint(self, state_dict, filepath):
        torch.save(state_dict, filepath)

    def get_paths(self):
        return self.best_paths