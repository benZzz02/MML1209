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

class MMLSurgAdaptTrainer():
    def __init__(self, distributed=False, gpu_id=0) -> None:
        super().__init__()
        self.distributed = distributed
        self.gpu_id = gpu_id

        clip_model, _ = load_clip_model()
        image_size = cfg.image_size

        train_preprocess = transforms.Compose([
            transforms.Resize((image_size,image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        val_preprocess = transforms.Compose([
            transforms.Resize((image_size,image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
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

        # [关键修改] 更新模型选择逻辑
        model_name = cfg.model
        if cfg.backbone == 'SurgVLP':
            self.model = SurgAVLP(clip_model, classnames, cfg.bert_path, cfg.vlp_weights)
        # 原始模型
        elif model_name == 'SurgAdapt': self.model = MMLSurgAdapt(classnames, clip_model)
        elif model_name == 'HSPNet': self.model = HSPNet(classnames, clip_model)
        elif model_name == 'VLPL': self.model = VLPL(classnames, clip_model)
        elif model_name == 'Resnet': self.model = Resnet(classnames, clip_model)
        elif model_name == 'ViT': self.model = ViT(classnames, clip_model)
        elif model_name == 'CLIP': self.model = CLIP_for_train(classnames, clip_model)
        elif model_name == 'CrossModel': self.model = CrossModel(classnames, clip_model) # 假设你也需要这个
        # CoOp 变体 (完全微调)
        elif model_name == 'SurgAdapt-CoOp': self.model = MMLSurgAdaptCoOp(classnames, clip_model)
        elif model_name == 'SurgAdapt-DualCoOp': self.model = MMLSurgAdaptDualCoOp(classnames, clip_model)
        elif model_name == 'SurgAdapt-CoCoOp': self.model = MMLSurgAdaptCoCoOp(classnames, clip_model)
       
        # CoOp 变体 (冻结骨干)
        elif model_name == 'SurgAdapt-CoOp-Frozen': self.model = MMLSurgAdaptCoOpFrozen(classnames, clip_model)
        elif model_name == 'SurgAdapt-DualCoOp-Frozen': self.model = MMLSurgAdaptDualCoOpFrozen(classnames, clip_model)
        elif model_name == 'SurgAdapt-CoCoOp-Frozen': self.model = MMLSurgAdaptCoCoOpFrozen(classnames, clip_model)
        elif model_name == 'SurgAdapt-CoOp-VisualFrozen': self.model = MMLSurgAdaptCoOpVisualFrozen(classnames, clip_model)
        
        elif model_name == 'CLIP-TextAttention': self.model = CLIP_TextAttention(classnames, clip_model)
        elif model_name == 'CLIP-TextAttention-CoOp': self.model = CLIP_TextAttentionCoOp(classnames, clip_model)
        elif model_name == 'CLIP-CoOp-LoRA': self.model = CLIPCoOpLoRA(classnames, clip_model)
        elif model_name == 'SCPNet': self.model = MMLSurgAdaptSCPNet(classnames, clip_model)
        
        else:
            raise NameError(f"Model '{model_name}' not recognized.")

        logger.info(f"Successfully initialized model: {model_name}")
        print(self.model)
        self.classnames = classnames

        self.model.cuda(self.gpu_id)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id, find_unused_parameters=True)

        ema_co = get_ema_co()
        logger.info(f"EMA CO: {ema_co}")
        self.model_unwrap = self.model.module if self.distributed else self.model
        self.ema = ModelEma(self.model_unwrap, ema_co)
        
        # [新增] 一致性损失相关配置
        self.use_consistency = getattr(cfg, 'use_consistency', False)
        # CLIP 归一化参数 (用于反归一化和重归一化)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).cuda(self.gpu_id)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).cuda(self.gpu_id)
        
        if self.use_consistency:
            logger.info("Consistency regularization is ENABLED")
        else:
            logger.info("Consistency regularization is DISABLED")
    
    def _apply_consistency_augmentation(self, image):
        """
        对归一化后的图像应用一致性增强
        
        Args:
            image: 归一化后的图像 tensor [B, C, H, W]
        
        Returns:
            增强后的图像 tensor [B, C, H, W]
        """
        # Augmentation constants
        RGB_TO_GRAY_WEIGHTS = [0.299, 0.587, 0.114]
        BRIGHTNESS_RANGE = 0.4
        BRIGHTNESS_MIN = 0.8
        
        with torch.no_grad():
            # 1. 反归一化到 [0, 1]
            image_denorm = image * self.clip_std + self.clip_mean
            image_denorm = torch.clamp(image_denorm, 0, 1)
            
            # 2. 应用随机增强
            batch_size = image_denorm.shape[0]
            image_aug = image_denorm.clone()
            
            # 随机水平翻转 (50% 概率) - vectorized
            flip_mask = torch.rand(batch_size) < 0.5
            if flip_mask.any():
                image_aug[flip_mask] = torch.flip(image_aug[flip_mask], dims=[2])
            
            # 随机亮度调整 (范围: 0.8 - 1.2)
            brightness_factor = torch.rand(batch_size, 1, 1, 1).cuda(self.gpu_id) * BRIGHTNESS_RANGE + BRIGHTNESS_MIN
            image_aug = image_aug * brightness_factor
            image_aug = torch.clamp(image_aug, 0, 1)
            
            # 随机灰度化 (20% 概率) - vectorized
            gray_mask = torch.rand(batch_size) < 0.2
            if gray_mask.any():
                # 转换为灰度图 (保持 3 通道) - vectorized computation
                gray = (RGB_TO_GRAY_WEIGHTS[0] * image_aug[:, 0] + 
                        RGB_TO_GRAY_WEIGHTS[1] * image_aug[:, 1] + 
                        RGB_TO_GRAY_WEIGHTS[2] * image_aug[:, 2])
                gray_3ch = gray.unsqueeze(1).repeat(1, 3, 1, 1)
                image_aug[gray_mask] = gray_3ch[gray_mask]
            
            # 3. 重新归一化
            image_aug = (image_aug - self.clip_mean) / self.clip_std
            
        return image_aug
    
    def train(self, input, target, criterion, epoch, epoch_i) -> torch.Tensor:
        image = input.cuda(non_blocking=True)
        
        # [新增] 检查是否需要生成增强数据用于一致性损失
        # 三个条件：1. use_consistency 开启  2. 训练模式  3. criterion 有 cons_weight 且 > 0
        need_augmentation = (
            self.use_consistency and 
            self.model.training and 
            hasattr(criterion, 'cons_weight') and 
            criterion.cons_weight > 0
        )
        
        # [修改] 智能判断：根据 Loss 的需求决定传什么
        if getattr(criterion, 'needs_features', False):
            # --- 分支 A: 针对 BBAM (需要特征) ---
            with autocast():
                # 使用 self.model_unwrap 统一处理 DDP 和 单卡情况
                # 确保 input 类型正确
                features = self.model_unwrap.image_encoder(image.type(self.model_unwrap.dtype))
                
                # 3. 归一化 (BBAM 必须步骤)
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
                
            # 4. 计算 Loss (传入特征)
            loss, _ = criterion(features, target, epoch)
            
        else:
            # --- 分支 B: 针对普通 Loss ---
            if need_augmentation:
                # 生成增强图像
                image_aug = self._apply_consistency_augmentation(image)
                # 拼接原图和增强图：[2*B, C, H, W]
                combined_image = torch.cat([image, image_aug], dim=0)
                
                with autocast():
                    # 模型前向传播，得到 [2*B, num_classes] 的输出
                    output = self.model(combined_image).float()
                
                # 损失函数内部会自动检测 logits.shape[0] == 2 * batch_size 并计算一致性损失
                loss, _ = criterion(output, target, epoch)
            else:
                # 不需要增强，正常前向传播
                with autocast():
                    output = self.model(image).float()
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