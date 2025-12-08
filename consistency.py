"""
Consistency Regularization 模块
包含图像增强类和一致性损失包装器
"""
import torch
import torch.nn as nn
import random


class ConsistencyAugmentor:
    """
    CPU 上的图像增强器，用于一致性正则化。
    对已归一化的图像进行增强，包括：水平翻转、亮度调整、灰度化
    """
    
    # CLIP 归一化参数
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        brightness_range: float = 0.4,
        brightness_min: float = 0.8,
        gray_prob: float = 0.2
    ):
        """
        Args:
            flip_prob: 水平翻转概率
            brightness_range: 亮度调整范围
            brightness_min: 亮度最小值
            gray_prob: 灰度化概率
        """
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.brightness_min = brightness_min
        self.gray_prob = gray_prob
        
        # 预计算归一化参数
        self.mean = torch.tensor(self. CLIP_MEAN).view(3, 1, 1)
        self.std = torch.tensor(self.CLIP_STD).view(3, 1, 1)
    
    def __call__(self, image: torch.Tensor) -> torch. Tensor:
        """
        对单张归一化后的图像应用增强
        
        Args:
            image: [C, H, W] 归一化后的图像
        
        Returns:
            增强后的图像 [C, H, W]
        """
        # 确保在 CPU 上操作
        device = image.device
        image = image.cpu()
        
        # 1. 反归一化到 [0, 1]
        image_denorm = image * self.std + self.mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        # 2. 随机水平翻转
        if random.random() < self. flip_prob:
            image_denorm = torch.flip(image_denorm, dims=[2])  # W 维度
        
        # 3. 随机亮度调整
        brightness_factor = self.brightness_min + random.random() * self.brightness_range
        image_denorm = image_denorm * brightness_factor
        image_denorm = torch.clamp(image_denorm, 0, 1)
        
        # 4. 随机灰度化
        if random.random() < self.gray_prob:
            gray = 0.299 * image_denorm[0] + 0.587 * image_denorm[1] + 0.114 * image_denorm[2]
            image_denorm = gray. unsqueeze(0).expand(3, -1, -1)
        
        # 5. 重新归一化
        image_aug = (image_denorm - self.mean) / self.std
        
        return image_aug. to(device)
    
    def augment_batch(self, images: torch. Tensor) -> torch.Tensor:
        """
        对一个 batch 的图像应用增强
        
        Args:
            images: [B, C, H, W] 归一化后的图像
        
        Returns:
            增强后的图像 [B, C, H, W]
        """
        augmented = []
        for i in range(images.shape[0]):
            augmented.append(self(images[i]))
        return torch.stack(augmented, dim=0)


class ConsistencyLoss(nn.Module):
    """
    一致性损失计算模块
    计算原图预测和增强图预测之间的 MSE 损失
    """
    
    def __init__(self, weight: float = 20.0, temperature: float = 1.0):
        """
        Args:
            weight: 一致性损失权重
            temperature: 温度参数，用于 softening predictions
        """
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.mse = nn.MSELoss(reduction='sum')
    
    def forward(
        self, 
        logits_clean: torch.Tensor, 
        logits_aug: torch. Tensor
    ) -> torch.Tensor:
        """
        计算一致性损失
        
        Args:
            logits_clean: 原图的 logits [B, C]
            logits_aug: 增强图的 logits [B, C]
        
        Returns:
            一致性损失值
        """
        # 计算 soft predictions
        probs_clean = torch.sigmoid(logits_clean / self.temperature). detach()
        probs_aug = torch.sigmoid(logits_aug / self.temperature)
        
        # MSE 损失
        loss = self.mse(probs_aug, probs_clean) * self.weight
        
        return loss