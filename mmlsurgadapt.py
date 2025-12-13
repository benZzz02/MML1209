import torch
from torch.cuda.amp import autocast
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing, InterpolationMode
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import build_dataset
from log import logger
from utils import ModelEma, get_ema_co
from typing import Optional, List, Tuple, Dict
import torch.nn as nn
from config import cfg

# [关键修改] 清理并导入所有需要的模型
from model import (
    load_clip_model, MMLSurgAdapt, Resnet, ViT, CrossModel, CLIP_for_train, VLPL, HSPNet,
    MMLSurgAdaptCoOp, MMLSurgAdaptDualCoOp, MMLSurgAdaptCoCoOp,
    MMLSurgAdaptCoOpFrozen, MMLSurgAdaptDualCoOpFrozen, MMLSurgAdaptCoCoOpFrozen, CLIP_TextAttention,CLIP_TextAttentionCoOp,CLIPCoOpLoRA,MMLSurgAdaptSCPNet
)
from surgvlp import SurgAVLP, CBertViT
from consistency import ConsistencyAugmentor


class MMLSurgAdaptTrainer():
    def __init__(self, distributed=False, gpu_id=0) -> None:
        super().__init__()
        self.distributed = distributed
        self. gpu_id = gpu_id

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
        if cfg.val_sp: self. val_sp_loader = sp_loader
        if cfg.perform_init: self. init_train_loader, self.init_val_loader = init_train_loader, init_val_loader
        classnames = val_loader.dataset.labels()

        # 模型初始化（保持原有逻辑）
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
        elif model_name == 'SCPNet': self.model = MMLSurgAdaptSCPNet(classnames, clip_model)
        elif model_name == 'SCPNet_Plus' : self.model = MMLSurgAdaptSCPNet(classnames, clip_model)
        else:
            raise NameError(f"Model '{model_name}' not recognized.")

        logger.info(f"Successfully initialized model: {model_name}")
        print(self.model)
        self.classnames = classnames

        self.model.cuda(self.gpu_id)
        if self.distributed:
            self. model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id, find_unused_parameters=True)

        ema_co = get_ema_co()
        logger.info(f"EMA CO: {ema_co}")
        self.model_unwrap = self.model.module if self.distributed else self.model
        self. ema = ModelEma(self.model_unwrap, ema_co)
        
        # ========== 一致性损失配置 ==========
        self. use_consistency = getattr(cfg, 'use_consistency', False)
        
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
    
    def train(self, input, target, criterion, epoch, epoch_i) -> torch.Tensor:
        """
        训练一个 batch
        """
        image = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # 判断是否需要一致性增强
        need_consistency = (
            self.use_consistency and
            self.model.training and
            self.augmentor is not None and
            hasattr(criterion, 'cons_weight') and
            criterion.cons_weight > 0
        )
        
        if getattr(criterion, 'needs_features', False):
            # BBAM 等需要特征的损失函数
            with autocast():
                features = self.model_unwrap. image_encoder(image. type(self.model_unwrap.dtype))
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            loss, _ = criterion(features, target, epoch)
            
        else:
            if need_consistency:
                # 在 CPU 上生成增强图像
                image_aug = self. augmentor. augment_batch(image. cpu()). cuda(non_blocking=True)
                
                # ✅ 关键修复：使用 self.model（DDP 包装的模型）保持梯度同步
                # 但要确保 batch size 正确
                with autocast():
                    # 拼接后一起前向传播，然后再拆分
                    combined = torch.cat([image, image_aug], dim=0)
                    combined_output = self.model(combined).float()
                    
                    # 拆分输出
                    batch_size = image.shape[0]
                    output_clean = combined_output[:batch_size]
                    output_aug = combined_output[batch_size:]
                
                # 调用 consistency loss
                loss, _ = criterion(output_clean, output_aug, target, epoch)
            else:
                with autocast():
                    output = self.model(image).float()
                # 非 consistency 模式
                if hasattr(criterion, 'cons_weight'):
                    loss, _ = criterion(output, None, target, epoch)
                else:
                    loss, _ = criterion(output, target, epoch)
            
        return loss

class customAugment(RandAugment):
    def __init__(self, num_ops: int = 2, magnitude: int = 8, num_magnitude_bins: int = 31, interpolation: InterpolationMode = InterpolationMode.NEAREST, fill: Optional[List[float]] = None) -> None:
        super().__init__()
    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False), "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True), "TranslateX": (torch.linspace(0.0, 150.0/331.0*image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0/331.0*image_size[0], num_bins), True), "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True), "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True), "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "AutoContrast": (torch.tensor(0.0), False), "Equalize": (torch.tensor(0.0), False),
        }