import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
import torch.distributed as dist
from config import cfg

class SPLC(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label 
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function, 
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions. 
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss. 

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(
        self,
        tau: float = 0.6,
        change_epoch: int = 1,
        margin: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits - self.margin, logits)

        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1,dtype=targets.dtype).cuda(), targets)

        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight
        #loss *= pt

        return loss.sum(), targets
    
class SPLC_WAN(nn.Module):
    r""" SPLC loss as described in the paper "Simple Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        &L_{SPLC}^+ = loss^+(p)
        &L_{SPLC}^- = \mathbb{I}(p\leq \tau)loss^-(p) + (1-\mathbb{I}(p\leq \tau))loss^+(p)

    where :math:'\tau' is a threshold to identify missing label 
          :math:`$\mathbb{I}(\cdot)\in\{0,1\}$` is the indicator function, 
          :math: $loss^+(\cdot), loss^-(\cdot)$ refer to loss functions for positives and negatives, respectively.

    .. note::
        SPLC can be combinded with various multi-label loss functions. 
        SPLC performs best combined with Focal margin loss in our paper. Code of SPLC with Focal margin loss is released here.
        Since the first epoch can recall few missing labels with high precision, SPLC can be used ater the first epoch.
        Sigmoid will be done in loss. 

    Args:
        tau (float): threshold value. Default: 0.6
        change_epoch (int): which epoch to combine SPLC. Default: 1
        margin (float): Margin value. Default: 1
        gamma (float): Hard mining value. Default: 2
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``

        """

    def __init__(
        self,
        tau: float = 0.6,
        change_epoch: int = 1,
        margin: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        super(SPLC, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where(targets == 1, logits - self.margin, logits)

        # SPLC missing label correction
        if epoch >= self.change_epoch:
            targets = torch.where(
                torch.sigmoid(logits) > self.tau,
                torch.tensor(1,dtype=targets.dtype).cuda(), targets)

        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred)**self.gamma * targets + (1/109) * (1 - targets)
        #focal_weight = pt**self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        #loss *= focal_weight
        loss *= pt

        return loss.sum(), targets
    
class GRLoss(nn.Module):

    def __init__(
            self,
            beta: list = [0,2,-2,-2],
            alpha: list = [0.5,2,0.8,0.5],
            q: list = [0.01,1],
    ) -> None:
        super(GRLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.q = q

    def neg_log(self,x):
        return (- torch.log(x + 1e-7))

    def loss1(self,x,q):
        return (1 - torch.pow(x, q)) / q

    def loss2(self,x,q):
        return (1 - torch.pow(1-x, q)) / q
    
    def K_function(self,preds,epoch):
        w_0,w_max,b_0,b_max=self.beta
        w=w_0+(w_max-w_0)*epoch/(cfg.epochs)
        b=b_0+(b_max-b_0)*epoch/(cfg.epochs)
        return 1 / (1 + torch.exp(-(w * preds + b)))
    
    def V_function(self,preds,epoch):
        mu_0,sigma_0,mu_max,sigma_max=self.alpha
        mu=mu_0+(mu_max-mu_0)*epoch/(cfg.epochs)
        sigma=sigma_0+(sigma_max-sigma_0)*epoch/(cfg.epochs)
        return torch.exp(-0.5 * ((preds - mu) / sigma) ** 2)  
    
    def forward(self,preds : torch.Tensor,label : torch.Tensor, epoch):
        preds = torch.sigmoid(preds)
        K = self.K_function(preds,epoch)
        V = self.V_function(preds,epoch)
        q2,q3=self.q
        loss_mtx = torch.zeros_like(preds)
        loss_mtx[label == 1]=self.neg_log(preds[label == 1])
        loss_mtx[label == 0]=V[label == 0]*(K[label == 0]*self.loss1(preds[label == 0],q2)+(1-K[label == 0])*self.loss2(preds[label == 0],q3))
        main_loss=loss_mtx.sum()
        return main_loss, label
    

class Hill(nn.Module):
    r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "

    .. math::
        Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2 

    where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives,
          : math:`m` is a margin parameter, 
          : math:`\gamma` is a commonly used value same as Focal loss.

    .. note::
        Sigmoid will be done in loss. 

    Args:
        lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.)
        margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result.)
        gamma (float): Commonly used value same as Focal loss. Default: 2

    """

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets, epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        # Hill loss calculation
        los_pos = targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        return loss.sum(), targets
    
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y, epoch):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum(), y
    
class WAN(nn.Module):

    def __init__(
        self,
        gamma: float = (1/109),
    ) -> None:
        super(WAN, self).__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """

        los_pos = targets * F.logsigmoid(logits)
        los_neg = self.gamma * (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)

        return loss.sum(), targets
    
class iWAN(nn.Module):

    def __init__(
        self,
        gamma: float = -(1/109),
        p: float = 0.5
    ) -> None:
        super(iWAN, self).__init__()
        self.gamma = gamma
        self.p = p
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        if epoch < 2:
            return self.bce(logits,targets), targets

        los_pos = targets * F.logsigmoid(logits)
        los_neg = self.gamma * (1/self.p) * (1 - targets) * (torch.sigmoid(logits)**self.p)

        loss = - (los_pos - los_neg)

        return 0.5*loss.sum(), targets
    
class G_AN(nn.Module):

    def __init__(
        self,
        gamma: float = -(1/109),
        q: float = 0.5
    ) -> None:
        super(G_AN, self).__init__()
        self.gamma = gamma
        self.q = q
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        if epoch < 2:
            return self.bce(logits,targets), targets

        los_pos = targets * F.logsigmoid(logits)
        los_neg = self.gamma * (1/self.q) * (1 - targets) * (1-(1-torch.sigmoid(logits))**self.q)
        loss = -(los_pos - los_neg)

        return 0.5*loss.sum(), targets

class VLPL_Loss(nn.Module):

    def __init__(
        self,
        theta: float = 0.3,
        delta: float = 0.1,
        alpha: float = 0.2,
        beta: float = 0.7,
        gamma : float = 0.0,
        rho1: float = 0.9,
        rho2 : float = 0.1, 
        num_classes: int = 110,
        warmup_epoch: int = 0
    ) -> None:
        super(VLPL_Loss, self).__init__()
        self.theta = theta
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho1 = rho1
        self.rho2 = rho2
        self.ncls = num_classes
        self.warmup_epoch = warmup_epoch
        # self.count = 0
    
    def neg_log(self,v):
        return - torch.log(v + 1e-7)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch):

        preds = torch.sigmoid(logits)
        pseudolabels = torch.zeros_like(preds).cuda()
        pseudolabels = torch.where(preds > self.theta,
                torch.tensor(1,dtype=pseudolabels.dtype).cuda(), pseudolabels)

        k = int(self.delta * self.ncls)
        if k > 0:
            _, lowest_k_indices = torch.topk(preds, k, largest=False)
            row_indices = torch.full(lowest_k_indices.shape,-1,dtype=pseudolabels.dtype).cuda()
            pseudolabels.scatter_(1,lowest_k_indices,row_indices)

        pseudo_pos_mask = (pseudolabels == 1).float()
        pseudo_neg_mask = (pseudolabels == -1).float()
        pseudo_unk_mask = (pseudolabels == 0).float()

        loss_positive = targets * self.neg_log(preds)
        loss_neg = - (1-targets) * self.alpha * (preds*self.neg_log(preds)+ (1-preds)*self.neg_log(1-preds))
        loss_pseudounk = - (1-targets) * pseudo_unk_mask * self.alpha * (preds*self.neg_log(preds)+ (1-preds)*self.neg_log(1-preds))
        loss_pseudopos = (1-targets) * pseudo_pos_mask * self.beta * ((1-self.rho1)*self.neg_log(1-preds)+self.rho1*self.neg_log(preds))
        loss_pseudoneg = (1-targets) * pseudo_neg_mask * self.gamma * ((1-self.rho2)*self.neg_log(1-preds)+self.rho2*self.neg_log(preds))

        if epoch > self.warmup_epoch:
            loss = loss_positive + loss_pseudounk + loss_pseudopos + loss_pseudoneg
        else:
            loss = loss_positive + loss_neg

        return loss.sum(), targets

class Modified_VLPL(nn.Module):

    def __init__(
        self,
        alpha: float = 0.2,
        num_classes: int = 110
    ) -> None:
        super(Modified_VLPL, self).__init__()
        self.alpha = alpha
        self.ncls = num_classes
    
    def neg_log(self,v):
        return - torch.log(v + 1e-7)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch):

        preds = torch.sigmoid(logits)

        loss_positive = targets * self.neg_log(preds)
        loss_neg = - (1-targets) * self.alpha * (preds*self.neg_log(preds)+ (1-preds)*self.neg_log(1-preds))

        loss = loss_positive + loss_neg

        return loss.sum(), targets
    
class LL(nn.Module):

    def __init__(
        self,
        delta_rel: float = 0.05,
        scheme: str = 'LL-Ct'
    ) -> None:
        super(LL, self).__init__()
        self.scheme = scheme
        self.delta_rel = delta_rel

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        unobserved_mask = (targets == 0)

        assert torch.min(targets) >= 0
        loss_matrix = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        corrected_loss_matrix = F.binary_cross_entropy_with_logits(logits, torch.logical_not(targets).float(), reduction='none')

        if epoch == 0:
            final_loss_matrix = loss_matrix
        else:
            clean_rate = 1 - epoch*self.delta_rel
            k = math.ceil(batch_size * num_classes * (1-clean_rate))
        
            unobserved_loss = unobserved_mask.bool() * loss_matrix
            topk = torch.topk(unobserved_loss.flatten(), k)
            topk_lossvalue = topk.values[-1]
            if self.scheme == 'LL-Ct':
                final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)
            else:
                zero_loss_matrix = torch.zeros_like(loss_matrix)
                final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, zero_loss_matrix)

        main_loss = final_loss_matrix.mean()
        
        return main_loss, targets
    
class Weighted_Hill(nn.Module):

    def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
        super(Weighted_Hill, self).__init__()
        self.lamb = lamb
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction
        self.task_w = self.get_task_weights(16112,6960,72815)
        assert(self.task_w.shape == (110,))

    def get_task_weights(self,n1,n2,n3):
        sigma_inv_ni = 1/n1 + 1/n2 + 1/n3
        w1 = (1/n1) / sigma_inv_ni
        w2 = (1/n2) / sigma_inv_ni
        w3 = (1/n3) / sigma_inv_ni
        t1 = torch.full((7,),w1)
        t2 = torch.full((3,),w2)
        t3 = torch.full((100,),w3)
        res = torch.cat((t1,t2,t3),dim=0)
        return res

    def forward(self, logits, targets, epoch):
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`

        Returns:
            torch.Tensor: loss
        """

        # Calculating predicted probability
        logits_margin = logits - self.margin
        pred_pos = torch.sigmoid(logits_margin)
        pred_neg = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred_pos) * targets + (1 - targets)
        focal_weight = pt ** self.gamma

        task_w = self.task_w.to(logits.device)

        # Hill loss calculation
        los_pos = task_w * targets * torch.log(pred_pos)
        los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        return loss.sum(), targets
    
LOG_EPSILON = 1e-7

# --- 辅助函数：需确保在 losses.py 顶部定义 ---
def neg_log(x: torch.Tensor) -> torch.Tensor:
    return -torch.log(x + LOG_EPSILON)

def loss1(x: torch.Tensor, q: float) -> torch.Tensor:
    return (1 - torch.pow(x, q)) / q

def loss2(x: torch.Tensor, q: float) -> torch.Tensor:
    return (1 - torch.pow(1 - x, q)) / q

def expected_positive_regularizer(preds: torch.Tensor, expected_num_pos: float, num_classes: int, norm: str = '2') -> torch.Tensor:
    if norm == '2':
        reg_val = (preds.sum(1).mean(0) - expected_num_pos)**2
    elif norm == '1':
        reg_val = torch.abs(preds.sum(1).mean(0) - expected_num_pos)
    else:
        raise NotImplementedError
    return reg_val / (num_classes ** 2)

# --- GPRLoss 修正版 ---
class GPRLoss(nn.Module):
    r""" 完整的 GPR Loss (Generalized Pattern Regularization Loss) 实现。
         已修复原地修改 (Inplace Modification) 导致的 Autograd 错误。
    """
    def __init__(
            self,
            beta: list = [0, 2, -2, -2],
            alpha: list = [0.5, 2, 0.8, 0.5],
            q: list = [0.01, 1],
            lam: list = [0.8, 0.3],
            rho: float = 0.9,
            reg: float = 0.001,
            use_pl: bool = True,
            pl_epoch_start: int = 1,
            expected_num_pos: float = 1.0,
            num_classes: int = 110,
            pl_pos_thresh: float = 0.5,
            pl_neg_thresh: float = 0.1,
    ) -> None:
        super(GPRLoss, self).__init__()
        self.beta = beta
        self.alpha = alpha
        self.q = q
        self.lam_1 = lam[0]
        self.lam_2 = lam[1]
        self.rho = rho
        self.reg = reg
        self.use_pl = use_pl
        self.pl_epoch_start = pl_epoch_start
        self.expected_num_pos = expected_num_pos
        self.num_classes = num_classes
        self.pl_pos_thresh = pl_pos_thresh
        self.pl_neg_thresh = pl_neg_thresh

    # K_function, V_function, neg_log, loss1, loss2 保持不变 (假设它们定义在类内部或外部)
    # 为了简洁，这里省略了 K_function, V_function, neg_log, loss1, loss2 的定义，
    # 请确保它们在您的实际文件中存在。

    def K_function(self,preds,epoch):
        w_0, w_max, b_0, b_max = self.beta
        w = w_0 + (w_max - w_0) * epoch / cfg.epochs
        b = b_0 + (b_max - b_0) * epoch / cfg.epochs
        return 1 / (1 + torch.exp(-(w * preds + b)))
    
    def V_function(self,preds,epoch):
        mu_0, sigma_0, mu_max, sigma_max = self.alpha
        mu = mu_0 + (mu_max - mu_0) * epoch / cfg.epochs
        sigma = sigma_0 + (sigma_max - sigma_0) * epoch / cfg.epochs
        return torch.exp(-0.5 * ((preds - mu) / sigma) ** 2)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: int):
        
        preds = torch.sigmoid(logits)
        K = self.K_function(preds, epoch)
        V = self.V_function(preds, epoch)
        q2, q3 = self.q
        device = targets.device
        
        # --- 核心修复：创建 V 的克隆，避免原地修改 ---
        # V_modified 是用于伪标签加权和 GPL 损失的权重张量
        V_modified = V.clone() 

        # 1. 伪标签生成 (使用您的阈值逻辑)
        pseudo_labels = torch.zeros_like(targets, dtype=targets.dtype, device=device)
        pseudo_labels[preds > self.pl_pos_thresh] = torch.tensor(1, dtype=targets.dtype, device=device)
        pseudo_labels[preds < self.pl_neg_thresh] = torch.tensor(-1, dtype=targets.dtype, device=device)
        
        # final_labels: 观测正样本 (targets=1) 优先，缺失样本使用伪标签
        final_labels = torch.where(targets == 1, targets, pseudo_labels)

        loss_mtx = torch.zeros_like(preds)
        
        # 2. 已知正样本损失
        loss_mtx[targets == 1] = self.neg_log(preds[targets == 1])

        # 3. 负样本基础损失 (GPL)
        # 应用于所有最终被视为负样本的 (final_labels == 0) 项。
        mask_gpl = (final_labels == 0)
        loss_mtx[mask_gpl] = V_modified[mask_gpl] * (
            K[mask_gpl] * self.loss1(preds[mask_gpl], q2) + 
            (1 - K[mask_gpl]) * self.loss2(preds[mask_gpl], q3)
        )

        # 4. 伪标签修正 (如果启用)
        if self.use_pl and epoch >= self.pl_epoch_start:
            
            # 4.1. 正伪标签 (Was missing, PL=1) - 覆盖 GPL 的结果
            mask_pos_pl = (targets == 0) & (final_labels == 1)
            if mask_pos_pl.any():
                # V 的范围修正：在 V_modified 上操作
                V_vals = V_modified[mask_pos_pl]
                V_vals = torch.where(V_vals > 1 - self.lam_1, V_vals, 1 - self.lam_1)
                V_vals = torch.where(V_vals < 1 - self.lam_2, V_vals, 1 - self.lam_2)
                
                # ⬅️ 安全的原地修改：修改 V_modified 的切片，而不是 V
                V_modified[mask_pos_pl] = V_vals 

                # 损失计算 (加权 BCE)
                loss_mtx[mask_pos_pl] = (1 - V_modified[mask_pos_pl]) * (
                    self.rho * self.neg_log(preds[mask_pos_pl]) + 
                    (1 - self.rho) * self.neg_log(1 - preds[mask_pos_pl])
                )
            
            # 4.2. 负伪标签 (Was missing, PL=-1) - 覆盖 GPL 的结果
            mask_neg_pl = (targets == 0) & (final_labels == -1)
            if mask_neg_pl.any():
                # ⬅️ 安全的修改：使用 V_modified
                loss_mtx[mask_neg_pl] = V_modified[mask_neg_pl] * self.neg_log(1 - preds[mask_neg_pl])
        
        # 5. 最终损失和正则化项
        main_loss = loss_mtx.mean()
        
        # 添加正则化项 (EPR)
        if self.reg > 0:
            main_loss += self.reg * expected_positive_regularizer(preds, self.expected_num_pos, self.num_classes)

        return main_loss, targets
    r""" 完整的 GPR Loss (Generalized Pattern Regularization Loss) 实现。
         所有超参数均通过 __init__ 传入，以便通过 YAML 文件进行控制。
         
         注意: 此实现内部基于预测值 (logits) 生成伪标签 (PL)，
         这与您提供的草稿逻辑一致，但您可以通过调整 pl_pos_thresh 和 pl_neg_thresh 
         来控制这个行为。
    """
    def __init__(
            self,
            # GPR Paramters (用于 K_function 和 V_function)
            beta: list = [0, 2, -2, -2],
            alpha: list = [0.5, 2, 0.8, 0.5],
            q: list = [0.01, 1],
            
            # GPR Paramters (用于损失计算和正则化)
            lam: list = [0.8, 0.3],     # [lam_1, lam_2] for V clamping. e.g., [0.8, 0.3] -> V bounds in [0.2, 0.7]
            rho: float = 0.9,           # Positive PL loss weighting (rho * neg_log(p) + (1-rho) * neg_log(1-p))
            reg: float = 0.001,
            use_pl: bool = True,
            pl_epoch_start: int = 1,
            expected_num_pos: float = 1.0,
            num_classes: int = 110,
            
            # 内部伪标签生成阈值 (如果不需要外部传入伪标签)
            pl_pos_thresh: float = 0.5,
            pl_neg_thresh: float = 0.1,
    ) -> None:
        super(GPRLoss, self).__init__()
        
        # K/V function parameters
        self.beta = beta
        self.alpha = alpha
        self.q = q
        
        # PL and Regularization parameters
        self.lam_1 = lam[0]
        self.lam_2 = lam[1]
        self.rho = rho
        self.reg = reg
        self.use_pl = use_pl
        self.pl_epoch_start = pl_epoch_start
        self.expected_num_pos = expected_num_pos
        self.num_classes = num_classes
        self.pl_pos_thresh = pl_pos_thresh
        self.pl_neg_thresh = pl_neg_thresh
        
        # 构造内部 P 字典 (主要用于 EPR，与原始实现保持一致)
        self.P_dict = {
            'expected_num_pos': self.expected_num_pos,
            'num_classes': self.num_classes,
        }

    # --- 辅助函数 (Internal Helpers) ---
    def neg_log(self, x: torch.Tensor) -> torch.Tensor:
        """ -log(x) """
        return -torch.log(x + LOG_EPSILON)

    def loss1(self, x: torch.Tensor, q: float) -> torch.Tensor:
        """ (1 - x^q) / q """
        return (1 - torch.pow(x, q)) / q

    def loss2(self, x: torch.Tensor, q: float) -> torch.Tensor:
        """ (1 - (1-x)^q) / q """
        return (1 - torch.pow(1 - x, q)) / q
    
    def expected_positive_regularizer(self, preds: torch.Tensor, norm: str = '2') -> torch.Tensor:
        """ 计算期望正样本数的正则项 (Expected Positive Regularizer) """
        # 使用 self.P_dict 中的参数
        if norm == '2':
            reg_val = (preds.sum(1).mean(0) - self.expected_num_pos)**2
        elif norm == '1':
            reg_val = torch.abs(preds.sum(1).mean(0) - self.expected_num_pos) # 支持 norm='1' 的逻辑
        else:
             raise NotImplementedError
        return reg_val / (self.num_classes ** 2)
    
    # --- K/V Functions ---
    def K_function(self, preds: torch.Tensor, epoch: int) -> torch.Tensor:
        w_0, w_max, b_0, b_max = self.beta
        # 假设 cfg.epochs 可用
        w = w_0 + (w_max - w_0) * epoch / (cfg.epochs)
        b = b_0 + (b_max - b_0) * epoch / (cfg.epochs)
        return 1 / (1 + torch.exp(-(w * preds + b)))
    
    def V_function(self, preds: torch.Tensor, epoch: int) -> torch.Tensor:
        mu_0, sigma_0, mu_max, sigma_max = self.alpha
        # 假设 cfg.epochs 可用
        mu = mu_0 + (mu_max - mu_0) * epoch / (cfg.epochs)
        sigma = sigma_0 + (sigma_max - sigma_0) * epoch / (cfg.epochs)
        return torch.exp(-0.5 * ((preds - mu) / sigma) ** 2)  
    
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: int):
        # -----------------
        # 1. 初始化和 K/V 计算
        # -----------------
        preds = torch.sigmoid(logits)
        K = self.K_function(preds, epoch)
        V = self.V_function(preds, epoch)
        q2, q3 = self.q
        device = targets.device

        # 1.1. 内部伪标签生成 (如果需要外部传入，则需要修改此处的 `forward` 签名和此逻辑)
        # 修正: 确保生成的张量在正确的设备上
        pseudo_labels = torch.zeros_like(targets, dtype=targets.dtype, device=device)
        pseudo_labels[preds > self.pl_pos_thresh] = torch.tensor(1, dtype=targets.dtype, device=device)
        pseudo_labels[preds < self.pl_neg_thresh] = torch.tensor(-1, dtype=targets.dtype, device=device)
        
        # 1.2. final_labels: 观测正样本 (targets=1) 优先，缺失样本使用伪标签
        # targets (label_vec) == 0 表示标签缺失
        final_labels = torch.where(targets == 1, targets, pseudo_labels)

        loss_mtx = torch.zeros_like(preds)
        
        # -----------------
        # 2. 损失计算
        # -----------------
        
        # 2.1. 已知正样本损失 (targets == 1)
        # 修正: 使用 targets == 1 筛选，与您上传的原始 GPR_loss 逻辑一致
        loss_mtx[targets == 1] = self.neg_log(preds[targets == 1])

        # 2.2. 负样本基础损失 (Generalized Partial Loss - GPL)
        # 应用于所有最终被视为负样本的 (final_labels == 0) 项，包含观测负样本和未被 PL 修正的缺失样本。
        mask_gpl = (final_labels == 0)
        loss_mtx[mask_gpl] = V[mask_gpl] * (
            K[mask_gpl] * self.loss1(preds[mask_gpl], q2) + 
            (1 - K[mask_gpl]) * self.loss2(preds[mask_gpl], q3)
        )

        # 2.3. 伪标签修正 (如果启用)
        if self.use_pl and epoch >= self.pl_epoch_start:
            # 2.3.1. 正伪标签 (Was missing, PL=1) - 这部分会覆盖 GPL 的结果
            mask_pos_pl = (targets == 0) & (final_labels == 1)
            
            if mask_pos_pl.any():
                # V 的范围修正 (1-lam_1 为下限，1-lam_2 为上限)
                V_vals = V[mask_pos_pl]
                V_vals = torch.where(V_vals > 1 - self.lam_1, V_vals, 1 - self.lam_1)
                V_vals = torch.where(V_vals < 1 - self.lam_2, V_vals, 1 - self.lam_2)
                V[mask_pos_pl] = V_vals # 写回 V

                # 损失计算 (加权 BCE)
                loss_mtx[mask_pos_pl] = (1 - V[mask_pos_pl]) * (
                    self.rho * self.neg_log(preds[mask_pos_pl]) + 
                    (1 - self.rho) * self.neg_log(1 - preds[mask_pos_pl])
                )
            
            # 2.3.2. 负伪标签 (Was missing, PL=-1) - 这部分也会覆盖 GPL 的结果
            mask_neg_pl = (targets == 0) & (final_labels == -1)
            
            if mask_neg_pl.any():
                loss_mtx[mask_neg_pl] = V[mask_neg_pl] * self.neg_log(1 - preds[mask_neg_pl])

        # -----------------
        # 3. 最终损失和正则化项
        # -----------------
        main_loss = loss_mtx.mean()
        
        # 添加正则化项 (EPR)
        if self.reg > 0:
            main_loss += self.reg * self.expected_positive_regularizer(preds)

        # 返回总损失和 targets
        return main_loss, targets
    r""" 完整的 GPR Loss (Generalized Pattern Regularization Loss) 实现。
         该实现集成了 K/V 函数、PL 修正、以及 EPR 正则化项。
    """
    def __init__(
            self,
            # GPR Paramters (用于 K_function 和 V_function)
            beta: list = [0, 2, -2, -2],
            alpha: list = [0.5, 2, 0.8, 0.5],
            q: list = [0.01, 1],
            
            # GPR Paramters (用于损失计算和正则化)
            lam: list = [0.05, 0.15], # lam[0]是lam_1, lam[1]是lam_2 (注意命名与论文原文一致)
            rho: float = 0.5,       # P['rho']
            reg: float = 0.001,
            use_pl: bool = True,
            pl_epoch_start: int = 1,
            expected_num_pos: float = 3.0,
            num_classes: int = 110,
    ) -> None:
        super(GPRLoss, self).__init__()
        # K/V function parameters
        self.beta = beta
        self.alpha = alpha
        self.q = q
        
        # PL and Regularization parameters
        self.lam_1 = lam[0] # Note: lam_1 corresponds to the tighter bound (closer to 1) in the original paper's notation
        self.lam_2 = lam[1] # Note: lam_2 corresponds to the looser bound (closer to 0)
        self.rho = rho
        self.reg = reg
        self.use_pl = use_pl
        self.pl_epoch_start = pl_epoch_start
        self.expected_num_pos = expected_num_pos
        self.num_classes = num_classes

        # 构造内部 P 字典以简化代码 (主要用于 EPR)
        self.P_dict = {
            'expected_num_pos': self.expected_num_pos,
            'num_classes': self.num_classes,
            'lam_1': self.lam_1,
            'lam_2': self.lam_2,
            'rho': self.rho,
            'reg': self.reg,
            'use_pl': self.use_pl,
        }

    # --- 辅助函数 (Internal Helpers) ---
    def neg_log(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.log(x + 1e-7)

    def loss1(self, x: torch.Tensor, q: float) -> torch.Tensor:
        return (1 - torch.pow(x, q)) / q

    def loss2(self, x: torch.Tensor, q: float) -> torch.Tensor:
        return (1 - torch.pow(1 - x, q)) / q
    
    # --- K/V Functions ---
    def K_function(self, preds: torch.Tensor, epoch: int) -> torch.Tensor:
        w_0, w_max, b_0, b_max = self.beta
        # 假设 cfg.epochs 可用
        w = w_0 + (w_max - w_0) * epoch / (cfg.epochs)
        b = b_0 + (b_max - b_0) * epoch / (cfg.epochs)
        return 1 / (1 + torch.exp(-(w * preds + b)))
    
    def V_function(self, preds: torch.Tensor, epoch: int) -> torch.Tensor:
        mu_0, sigma_0, mu_max, sigma_max = self.alpha
        # 假设 cfg.epochs 可用
        mu = mu_0 + (mu_max - mu_0) * epoch / (cfg.epochs)
        sigma = sigma_0 + (sigma_max - sigma_0) * epoch / (cfg.epochs)
        return torch.exp(-0.5 * ((preds - mu) / sigma) ** 2)  

    def expected_positive_regularizer(self, preds: torch.Tensor, norm: str = '2') -> torch.Tensor:
        """ 计算期望正样本数的正则项 (Expected Positive Regularizer)"""
        if norm == '2':
            # 使用 self.expected_num_pos 和 self.num_classes
            reg_val = (preds.sum(1).mean(0) - self.expected_num_pos)**2
        else:
             # 如果 norm='1' 也需要支持，则添加逻辑
             reg_val = torch.abs(preds.sum(1).mean(0) - self.expected_num_pos) # 示例

        return reg_val / (self.num_classes ** 2)

    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch):
        # -----------------
        # 1. 初始化和 K/V 计算
        # -----------------
        preds = torch.sigmoid(logits)
        K = self.K_function(preds, epoch)
        V = self.V_function(preds, epoch)
        q2, q3 = self.q
        device = targets.device

        # --- 关键设计问题修正：伪标签的生成 ---
        # ⚠️ 警告：当前的 `forward` 设计内部生成了伪标签，使其依赖于固定的 0.5 和 0.1 阈值，
        # 而不是通过外部模块提供。这可能不是论文作者的本意，且灵活性较差。
        # 如果您的训练流程中有一个专门的模块生成伪标签，请修改 forward 签名以接收它。
        pseudo_labels = torch.zeros_like(targets)
        
        # 修正: 使用 targets.device 保证设备一致性
        pseudo_labels[preds > 0.5] = torch.tensor(1, dtype=targets.dtype, device=device)
        pseudo_labels[preds < 0.1] = torch.tensor(-1, dtype=targets.dtype, device=device)
        
        # final_labels: 观测正样本 (targets=1) 优先，缺失样本使用伪标签
        final_labels = torch.where(targets == 1, targets, pseudo_labels)

        loss_mtx = torch.zeros_like(preds)
        
        # -----------------
        # 2. 损失计算
        # -----------------
        
        # 2.1. 已知正样本损失 (targets == 1)
        loss_mtx[targets == 1] = self.neg_log(preds[targets == 1])

        # 2.2. 负样本基础损失 (Generalized Partial Loss - GPL)
        # 应用于所有最终被视为负样本的（final_labels == 0）项。
        mask_gpl = (final_labels == 0) # 修正: 简化掩码为 final_labels == 0
        loss_mtx[mask_gpl] = V[mask_gpl] * (
            K[mask_gpl] * self.loss1(preds[mask_gpl], q2) + 
            (1 - K[mask_gpl]) * self.loss2(preds[mask_gpl], q3)
        )

        # 2.3. 伪标签修正 (如果启用)
        if self.use_pl and epoch >= self.pl_epoch_start:
            # 2.3.1. 正伪标签 (Was missing, PL=1) - 这部分会覆盖 GPL 的结果
            mask_pos_pl = (targets == 0) & (final_labels == 1) # final_labels=1 意味着 PL=1
            
            if mask_pos_pl.any():
                # V 的范围修正 (1-lam_1 为下限，1-lam_2 为上限)
                V_vals = V[mask_pos_pl]
                V_vals = torch.where(V_vals > 1 - self.lam_1, V_vals, 1 - self.lam_1)
                V_vals = torch.where(V_vals < 1 - self.lam_2, V_vals, 1 - self.lam_2)
                V[mask_pos_pl] = V_vals # 写回 V

                # 损失计算
                loss_mtx[mask_pos_pl] = (1 - V[mask_pos_pl]) * (
                    self.rho * self.neg_log(preds[mask_pos_pl]) + 
                    (1 - self.rho) * self.neg_log(1 - preds[mask_pos_pl])
                )
            
            # 2.3.2. 负伪标签 (Was missing, PL=-1) - 这部分会覆盖 GPL 的结果
            mask_neg_pl = (targets == 0) & (final_labels == -1) # final_labels=-1 意味着 PL=-1
            
            if mask_neg_pl.any():
                loss_mtx[mask_neg_pl] = V[mask_neg_pl] * self.neg_log(1 - preds[mask_neg_pl])

        # -----------------
        # 3. 最终损失和正则化项
        # -----------------
        main_loss = loss_mtx.mean()
        
        # 添加正则化项
        if self.reg > 0:
            # 内部 P 字典包含所有 EPR 所需参数
            main_loss += self.reg * self.expected_positive_regularizer(preds)

        # 返回总损失和最终使用的标签 (通常是targets，但GPR的输出可能需要final_labels)
        # 这里返回targets保持API兼容性，但在GPR的内部逻辑中实际使用了final_labels
        return main_loss, targets
    



class BBAMLossVisual(nn.Module):
    def __init__(
        self,
        num_classes: int,
        # feat_dim: int,  <-- [懒加载] 不再需要手动指定维度
        s: float = 10.0,       # 必须与 model.py 中的 logit_scale 一致 (MMLSurgAdapt 是 10)
        m: float = 0.4,        # Margin
        start_epoch: int = 5,  # Warmup 结束的 epoch
        gamma: float = 0.99,   # 角度统计量 (Means/Vars) 的更新动量
        center_gamma: float = 0.9, # 视觉原型 (Centers) 的更新动量
        eps: float = 1e-7
    ) -> None:
        super(BBAMLossVisual, self).__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.start_epoch = start_epoch
        self.gamma = gamma
        self.center_gamma = center_gamma
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # [关键标记] 告诉 Trainer 这个 Loss 需要图像特征而不是 Logits
        self.needs_features = True 
        
        # --- 核心状态缓存 ---
        # 1. 视觉原型 (Centers): 设为 None，第一次 forward 时自动检测并初始化
        self.register_buffer('centers', None)
        self.register_buffer('centers_initialized', torch.tensor(0.0))

        # 2. 角度分布统计量 [2, C]: 存储正(1)/负(0)样本的角度均值和方差
        self.register_buffer('means', torch.zeros(2, num_classes))
        self.register_buffer('variances', torch.ones(2, num_classes))

    def _init_centers(self, feature_dim, device):
        """自动初始化 Centers 矩阵"""
        self.centers = torch.zeros(self.num_classes, feature_dim).to(device)

    def _update_centers(self, features, targets):
        """使用动量(EMA)更新视觉原型，支持 DDP"""
        # [自动初始化检查]
        if self.centers is None:
            self._init_centers(features.shape[1], features.device)

        # features: [B, D], targets: [B, C]
        pos_mask = (targets == 1).float()
        
        # 计算当前 Batch 的特征和
        batch_sum = torch.matmul(pos_mask.t(), features) # [C, D]
        batch_count = pos_mask.sum(0).unsqueeze(1)       # [C, 1]
        
        # --- DDP 同步 (关键) ---
        if dist.is_initialized():
            packed = torch.cat([batch_sum, batch_count], dim=1)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            batch_sum = packed[:, :-1]
            batch_count = packed[:, -1:]

        # 哪些类在这个(全局)Batch里有数据？
        has_data_mask = (batch_count > 0).float()
        
        # 计算当前 Batch 的平均中心
        current_centers = batch_sum / (batch_count + self.eps)
        current_centers = F.normalize(current_centers, p=2, dim=1)

        # --- 动量更新 ---
        if self.centers_initialized.item() < 0.5:
            # 初始化阶段：直接覆盖有数据的类
            self.centers = self.centers * (1 - has_data_mask) + current_centers * has_data_mask
            if has_data_mask.sum() > 0: 
                self.centers_initialized.fill_(1.0)
        else:
            # 稳定阶段：Centers = (1-alpha) * Old + alpha * New
            alpha = (1 - self.center_gamma) * has_data_mask
            self.centers = (1 - alpha) * self.centers + alpha * current_centers
            
        # 始终保持归一化
        self.centers = F.normalize(self.centers, p=2, dim=1)

    def _update_stats(self, theta, targets):
        """更新角度分布统计量 (均值/方差)，支持 DDP"""
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()
        
        # 本地统计
        num_pos = pos_mask.sum(0)
        num_neg = neg_mask.sum(0)
        sum_pos = (theta * pos_mask).sum(0)
        sum_neg = (theta * neg_mask).sum(0)
        # 方差近似计算：利用当前的全局 means
        diff_sq_pos = ((theta - self.means[1])**2 * pos_mask).sum(0)
        diff_sq_neg = ((theta - self.means[0])**2 * neg_mask).sum(0)

        # DDP 同步
        if dist.is_initialized():
            packed = torch.stack([num_pos, num_neg, sum_pos, sum_neg, diff_sq_pos, diff_sq_neg])
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            num_pos, num_neg, sum_pos, sum_neg, diff_sq_pos, diff_sq_neg = packed

        # EMA 更新
        has_pos = (num_pos > 0)
        has_neg = (num_neg > 0)
        
        denom_pos = num_pos.clamp(min=self.eps)
        denom_neg = num_neg.clamp(min=self.eps)

        if has_pos.any():
            batch_mean = sum_pos / denom_pos
            batch_var = diff_sq_pos / denom_pos
            self.means[1][has_pos] = self.gamma * self.means[1][has_pos] + (1 - self.gamma) * batch_mean[has_pos]
            self.variances[1][has_pos] = self.gamma * self.variances[1][has_pos] + (1 - self.gamma) * batch_var[has_pos]

        if has_neg.any():
            batch_mean = sum_neg / denom_neg
            batch_var = diff_sq_neg / denom_neg
            self.means[0][has_neg] = self.gamma * self.means[0][has_neg] + (1 - self.gamma) * batch_mean[has_neg]
            self.variances[0][has_neg] = self.gamma * self.variances[0][has_neg] + (1 - self.gamma) * batch_var[has_neg]

    def forward(self, image_features, targets, epoch):
        """
        Args:
            image_features: [B, D] 归一化后的图像特征
            targets: [B, C] 0/1 标签
        """
        # [自动初始化检查]
        if self.centers is None:
             self._init_centers(image_features.shape[1], image_features.device)

        # --- 1. 训练阶段：永远尝试更新 Center 和 Stats ---
        if self.training:
            with torch.no_grad():
                self._update_centers(image_features.detach(), targets)
                
                # 基于最新的 Center 计算角度
                cosine = torch.matmul(image_features, self.centers.t())
                cosine = torch.clamp(cosine, -1.0 + self.eps, 1.0 - self.eps)
                theta = torch.acos(cosine)
                
                self._update_stats(theta, targets)

        # --- 2. 计算 Loss ---
        
        # 重新计算 Cosine
        cosine_pred = torch.matmul(image_features, self.centers.t())
        
        if epoch < self.start_epoch:
            # Warmup: 标准 Cosine Loss
            logits = self.s * cosine_pred
            loss = self.bce(logits, targets)
        else:
            # BBAM: 方差平衡变换
            theta_pred = torch.acos(torch.clamp(cosine_pred, -1.0+self.eps, 1.0-self.eps))
            
            mean_var = self.variances.mean(dim=0)
            var_pos = self.variances[1] + self.eps
            var_neg = self.variances[0] + self.eps
            
            a_pos = torch.sqrt(mean_var / var_pos)
            b_pos = (1 - a_pos) * self.means[1]
            a_neg = torch.sqrt(mean_var / var_neg)
            b_neg = (1 - a_neg) * self.means[0]
            
            theta_trans_pos = a_pos * theta_pred + b_pos
            theta_trans_neg = a_neg * theta_pred + b_neg
            
            theta_final = torch.where(targets == 1, theta_trans_pos, theta_trans_neg)
            
            phi = torch.cos(theta_final)
            logits_bbam = self.s * (phi - targets * self.m)
            
            loss = self.bce(logits_bbam, targets)

        return loss.sum(), targets
   
class GCELoss(nn.Module):
    def __init__(self, q=0.7, warmup_epochs=5):
        super(GCELoss, self).__init__()
        self.q = q
        self.warmup_epochs = warmup_epochs
        
        # [修改] 将 Warmup Loss 改为 Hill
        # 这里的参数建议和你 yaml 配置里的一致 (lamb=1.5, margin=1.0 等)
        self.warmup_loss = Hill(lamb=1.5, margin=1.0, gamma=2.0)
        
        self.epsilon = 1e-5 

    def forward(self, logits, targets, epoch):
        # --- 阶段 1: Warmup (使用 Hill Loss) ---
        if epoch < self.warmup_epochs:
            # Hill Loss 的 forward 需要 epoch 参数，虽然它内部没怎么用，但为了接口一致
            loss, _ = self.warmup_loss(logits, targets, epoch)
            return loss, targets

        # --- 阶段 2: Robust Training (切换为 GCE) ---
        p = torch.sigmoid(logits)
        
        # 计算 p_true
        p_true = p * targets + (1 - p) * (1 - targets)
        p_true = p_true + self.epsilon
        
        # GCE 公式
        loss = (1.0 - torch.pow(p_true, self.q)) / self.q
        
        return loss.sum(), targets
    
class SCELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=110):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets, epoch):
        """
        logits: (Batch, Num_Classes) - 未经过 Sigmoid 的输出
        targets: (Batch, Num_Classes) - 0/1 标签 (包含伪标签)
        """
        # 1. CE 部分 (注重收敛)
        ce_loss = self.bce(logits, targets)

        # 2. RCE 部分 (注重抗噪)
        # 在多标签/二分类场景下，RCE 通常用 MAE (Mean Absolute Error) 近似替代
        # 以避免 log(0) 的数值问题，同时保持对噪声的鲁棒性（梯度有界）
        pred = torch.sigmoid(logits)
        mae_loss = torch.abs(pred - targets)

        # 3. 组合
        loss = self.alpha * ce_loss + self.beta * mae_loss

        return loss.sum(), targets