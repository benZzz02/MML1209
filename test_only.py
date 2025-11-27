import os
import sys
import torch
import torch.distributed as dist
from datetime import timedelta
from omegaconf import OmegaConf 

# 1. 导入项目配置和模块
from config import cfg
from mmlsurgadapt import MMLSurgAdaptTrainer
# 导入你不想修改的 train.py 中的 test 函数
from train import test 

def main():
    # ==========================================
    # 核心修复：不动 train.py，在这里注入缺失的参数
    # ==========================================
    print(f"[Fix] 正在为 cfg 注入 gpu_id 参数...")
    try:
        # 解锁 OmegaConf (防止它因为 strict mode 报错)
        OmegaConf.set_struct(cfg, False)
    except:
        pass
    
    # 获取当前 GPU ID 并赋值给 cfg.gpu_id
    # 这样 train.py 运行到第 736 行时就能取到值了
    if torch.cuda.is_available():
        cfg.gpu_id = torch.cuda.current_device()
    else:
        cfg.gpu_id = 0
        
    print(f"[Fix] 注入成功: cfg.gpu_id = {cfg.gpu_id}")
    # ==========================================


    # 2. 环境强制设置 (防止 cuDNN/nvrtc 报错)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except:
        pass

    # 3. 伪装单卡 DDP 环境 (防止死锁)
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12348' # 换个端口
    os.environ['LOCAL_RANK'] = '0'

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # 初始化进程组 (必须步骤)
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(minutes=60))

    # 4. 路径逻辑 (适配你的 round22/trial 结构)
    # 假设你的 checkpoints 位于: checkpoints/cholec_vals/round22/trial/...
    # train.py 的逻辑是拼接: cfg.checkpoint + "/" + dir + "/" + method + ...
    
    target_round = "round22"
    target_subdir = "trial"
    
    # 智能修正 checkpoint 根路径
    if target_round not in cfg.checkpoint:
        # 如果 config 里是 default 的，我们就把它拼成 .../round22
        cfg.checkpoint = os.path.join(cfg.checkpoint, target_round)
    
    print(f"Checkpoint Path: {cfg.checkpoint}")
    print(f"Target Subdir  : {target_subdir}")

    # 5. 初始化模型并测试
    trainer = MMLSurgAdaptTrainer()
    
    # 直接调用 train.py 的 test
    test(trainer, target_subdir)
    
    dist.destroy_process_group()
    print("测试全部完成。")

if __name__ == '__main__':
    main()