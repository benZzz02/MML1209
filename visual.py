import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import re

# ==========================================
# 标签定义 (你提供的完整列表)
# ==========================================
BOUNDARIES = [0, 7, 10, 110] 
RAW_PROMPTS = [
    # --- 7 Phases ---
    "of the phase Preparation", "of the phase Calot Triangle Dissection", "of the phase Clipping Cutting", "of the phase Gallbladder Dissection", "of the phase Gallbladder Retraction", "of the phase Cleaning Coagulation", "of the phase Gallbladder Packaging",
    # --- 3 Safe Views ---
    "of seeing two structures cystic duct and cystic artery", "of carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate", "of the lower part of the gallbladder divided from the liver bed to expose the cystic plate",
    # --- 100 Action Triplets ---
    "of the tool grasper performing the action dissect on the target cystic plate", "of the tool grasper performing the action dissect on the target gallbladder", "of the tool grasper performing the action dissect on the target omentum", "of the tool grasper performing the action grasp on the target cystic artery", "of the tool grasper performing the action grasp on the target cystic duct", "of the tool grasper performing the action grasp on the target cystic pedicle", "of the tool grasper performing the action grasp on the target cystic plate", "of the tool grasper performing the action grasp on the target gallbladder", "of the tool grasper performing the action grasp on the target gut", "of the tool grasper performing the action grasp on the target liver", "of the tool grasper performing the action grasp on the target omentum", "of the tool grasper performing the action grasp on the target peritoneum", "of the tool grasper performing the action grasp on the target specimen bag", "of the tool grasper performing the action pack on the target gallbladder", "of the tool grasper performing the action retract on the target cystic duct", "of the tool grasper performing the action retract on the target cystic pedicle", "of the tool grasper performing the action retract on the target cystic plate", "of the tool grasper performing the action retract on the target gallbladder", "of the tool grasper performing the action retract on the target gut", "of the tool grasper performing the action retract on the target liver", "of the tool grasper performing the action retract on the target omentum", "of the tool grasper performing the action retract on the target peritoneum",
    "of the tool bipolar performing the action coagulate on the target abdominal wall cavity", "of the tool bipolar performing the action coagulate on the target blood vessel", "of the tool bipolar performing the action coagulate on the target cystic artery", "of the tool bipolar performing the action coagulate on the target cystic duct", "of the tool bipolar performing the action coagulate on the target cystic pedicle", "of the tool bipolar performing the action coagulate on the target cystic plate", "of the tool bipolar performing the action coagulate on the target gallbladder", "of the tool bipolar performing the action coagulate on the target liver", "of the tool bipolar performing the action coagulate on the target omentum", "of the tool bipolar performing the action coagulate on the target peritoneum", "of the tool bipolar performing the action dissect on the target adhesion", "of the tool bipolar performing the action dissect on the target cystic artery", "of the tool bipolar performing the action dissect on the target cystic duct", "of the tool bipolar performing the action dissect on the target cystic plate", "of the tool bipolar performing the action dissect on the target gallbladder", "of the tool bipolar performing the action dissect on the target omentum", "of the tool bipolar performing the action grasp on the target cystic plate", "of the tool bipolar performing the action grasp on the target liver", "of the tool bipolar performing the action grasp on the target specimen bag", "of the tool bipolar performing the action retract on the target cystic duct", "of the tool bipolar performing the action retract on the target cystic pedicle", "of the tool bipolar performing the action retract on the target gallbladder", "of the tool bipolar performing the action retract on the target liver", "of the tool bipolar performing the action retract on the target omentum",
    "of the tool hook performing the action coagulate on the target blood vessel", "of the tool hook performing the action coagulate on the target cystic artery", "of the tool hook performing the action coagulate on the target cystic duct", "of the tool hook performing the action coagulate on the target cystic pedicle", "of the tool hook performing the action coagulate on the target cystic plate", "of the tool hook performing the action coagulate on the target gallbladder", "of the tool hook performing the action coagulate on the target liver", "of the tool hook performing the action coagulate on the target omentum", "of the tool hook performing the action cut on the target blood vessel", "of the tool hook performing the action cut on the target peritoneum", "of the tool hook performing the action dissect on the target blood vessel", "of the tool hook performing the action dissect on the target cystic artery", "of the tool hook performing the action dissect on the target cystic duct", "of the tool hook performing the action dissect on the target cystic plate", "of the tool hook performing the action dissect on the target gallbladder", "of the tool hook performing the action dissect on the target omentum", "of the tool hook performing the action dissect on the target peritoneum", "of the tool hook performing the action retract on the target gallbladder", "of the tool hook performing the action retract on the target liver",
    "of the tool scissors performing the action coagulate on the target omentum", "of the tool scissors performing the action cut on the target adhesion", "of the tool scissors performing the action cut on the target blood vessel", "of the tool scissors performing the action cut on the target cystic artery", "of the tool scissors performing the action cut on the target cystic duct", "of the tool scissors performing the action cut on the target cystic plate", "of the tool scissors performing the action cut on the target liver", "of the tool scissors performing the action cut on the target omentum", "of the tool scissors performing the action cut on the target peritoneum", "of the tool scissors performing the action dissect on the target cystic plate", "of the tool scissors performing the action dissect on the target gallbladder", "of the tool scissors performing the action dissect on the target omentum",
    "of the tool clipper performing the action clip on the target blood vessel", "of the tool clipper performing the action clip on the target cystic artery", "of the tool clipper performing the action clip on the target cystic duct", "of the tool clipper performing the action clip on the target cystic pedicle", "of the tool clipper performing the action clip on the target cystic plate",
    "of the tool irrigator performing the action aspirate on the target fluid", "of the tool irrigator performing the action dissect on the target cystic duct", "of the tool irrigator performing the action dissect on the target cystic pedicle", "of the tool irrigator performing the action dissect on the target cystic plate", "of the tool irrigator performing the action dissect on the target gallbladder", "of the tool irrigator performing the action dissect on the target omentum", "of the tool irrigator performing the action irrigate on the target abdominal wall cavity", "of the tool irrigator performing the action irrigate on the target cystic pedicle", "of the tool irrigator performing the action irrigate on the target liver", "of the tool irrigator performing the action retract on the target gallbladder", "of the tool irrigator performing the action retract on the target liver", "of the tool irrigator performing the action retract on the target omentum",
    "of the tool grasper performing the action null verb on the target null target", "of the tool bipolar performing the action null verb on the target null target", "of the tool hook performing the action null verb on the target null target", "of the tool scissors performing the action null verb on the target null target", "of the tool clipper performing the action null verb on the target null target", "of the tool irrigator performing the action null verb on the target null target"
]

def clean_labels(raw_list):
    """标签清洗，保持短小精悍"""
    clean_list = []
    for item in raw_list:
        if "of the phase" in item:
            l = item.replace("of the phase ", "").replace("Calot Triangle Dissection", "Calot Tri.").replace("Gallbladder", "GB")
            clean_list.append(l)
        elif "seeing two structures" in item: clean_list.append("View: 2 Structures")
        elif "hepatocystic triangle" in item: clean_list.append("View: Hep. Tri.")
        elif "divided from the liver bed" in item: clean_list.append("View: GB-Liver")
        else:
            match = re.search(r"tool (.*?) performing the action (.*?) on the target (.*)", item)
            if match:
                t, a, tg = match.groups()
                if "null" in a: a = "None"
                if "null" in tg: tg = "None"
                clean_list.append(f"{t.capitalize()}-{a}-{tg}")
            else:
                clean_list.append(item[:15])
    return clean_list

def print_top_correlations(matrix, names, top_k=20):
    """
    [数字版核心] 在控制台打印数值，证明相似度是存在的
    只关注 Phase (0-7) 与 Action (10-110) 之间的关系
    """
    print("\n" + "="*50)
    print(f"📊 数字分析: Phase vs Action 相关性 Top {top_k}")
    print("="*50)
    
    # 取出 Phase-Action 区域
    phase_action_block = matrix[0:7, 10:110]
    
    # 展平并获取索引
    flat_indices = np.argsort(phase_action_block.flatten())[::-1] # 降序
    
    count = 0
    for idx in flat_indices:
        if count >= top_k: break
        
        # 还原坐标
        phase_idx = idx // 100
        action_idx = (idx % 100) + 10 # 加上偏移量
        
        score = matrix[phase_idx, action_idx]
        if score < 0.001: continue # 忽略极小值
        
        p_name = names[phase_idx]
        a_name = names[action_idx]
        
        print(f"[{score:.4f}]  {p_name}  <==>  {a_name}")
        count += 1
    print("="*50 + "\n")

def plot_digital_astar(matrix, class_names, save_path):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()

    # --- 1. 自动计算合适的显示范围 ---
    # 移除对角线(1.0)后的最大值，用于设定颜色上限，防止被1.0拉低对比度
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    max_val_off_diag = matrix[mask].max()
    
    # 如果最大值很小(比如0.1)，就把上限设为0.12，这样0.1就会显示为大红色
    vmax = max(max_val_off_diag * 1.1, 0.05) 
    print(f"🎨 自动色彩范围: vmin=0.0, vmax={vmax:.4f} (以此增强可见度)")

    plt.figure(figsize=(26, 22))
    
    # --- 2. 绘制增强热力图 ---
    # 使用 'turbo' 或 'jet' 这种高对比度色谱
    ax = sns.heatmap(
        matrix, 
        cmap='turbo',    # 🌈 这种颜色对数值变化非常敏感
        vmin=0.0, 
        vmax=vmax,       # 动态上限
        square=False,
        xticklabels=class_names,  
        yticklabels=class_names,  
        cbar_kws={'label': f'Correlation Strength (Scaled 0-{vmax:.2f})', 'shrink': 0.6}
    )
    
    # 绘制分隔线
    for b in BOUNDARIES[1:-1]:
        plt.axvline(x=b, color='white', linestyle='--', linewidth=1.5)
        plt.axhline(y=b, color='white', linestyle='--', linewidth=1.5)

    # 坐标轴设置
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    # 区域大字标注
    plt.text(-3, 3.5, "Phase", va='center', ha='right', fontsize=20, fontweight='bold')
    plt.text(-3, 8.5, "View", va='center', ha='right', fontsize=20, fontweight='bold')
    plt.text(-3, 60, "Action", va='center', ha='right', fontsize=20, fontweight='bold')

    plt.title(f"A* Matrix (Max Similarity: {matrix.max():.4f})", fontsize=24, pad=20)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"✅ 高清数字版热力图已保存至: {save_path}")

def load_astar(ckpt_path):
    if not os.path.exists(ckpt_path):
        return None
    print(f"📂 加载: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    for k in state_dict.keys():
        if "A_star" in k:
            return state_dict[k]
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help="必须指定权重文件路径")
    parser.add_argument('--save', type=str, default="visual/digital_astar.png")
    args = parser.parse_args()

    # 1. 准备标签
    clean_names = clean_labels(RAW_PROMPTS)

    # 2. 加载数据
    a_star = load_astar(args.ckpt)

    if a_star is not None:
        # 3. 数字分析 (在终端看数字)
        print_top_correlations(a_star.cpu().numpy(), clean_names)
        
        # 4. 绘图
        plot_digital_astar(a_star, clean_names, args.save)
    else:
        print("❌ 未找到模型或 A_star 矩阵")