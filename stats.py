import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import re

# ================= 1. 配置与字体加载 =================
# 指定本地字体文件路径
FONT_PATH = "simhei.ttf" 

if os.path.exists(FONT_PATH):
    font_prop = FontProperties(fname=FONT_PATH)
    print(f"成功加载本地字体: {FONT_PATH}")
else:
    font_prop = None
    print(f"错误: 未在当前目录找到 {FONT_PATH}！中文将无法正常显示。")

plt.rcParams['axes.unicode_minus'] = False 

DATA_PATH = "/data/cholecdata"
LABEL_NAMES_FILE = 'cholec/cholec_labels.txt'
NUM_CLASSES = 110
SAVE_PATH = "train_dist_final_chinese.png"

# ================= 2. 完整全量翻译映射 =================
# 直接使用你提供的原始完整字符串作为 Key
FULL_LABEL_MAP = {
    # 阶段 Phase
    "of the phase Preparation": "准备阶段",
    "of the phase Calot Triangle Dissection": "胆囊三角解剖",
    "of the phase Clipping Cutting": "管路夹闭切断",
    "of the phase Gallbladder Dissection": "胆囊壁剥离",
    "of the phase Gallbladder Retraction": "胆囊提取",
    "of the phase Cleaning Coagulation": "冲洗与止血",
    "of the phase Gallbladder Packaging": "胆囊装袋",
    
    # 视图 View (修正后的三项)
    "of seeing two structures cystic duct and cystic artery": "观察到胆囊管与胆囊动脉",
    "of carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate": "CVS关键安全视野(胆囊三角解剖清晰)",
    "of the lower part of the gallbladder divided from the liver bed to expose the cystic plate": "胆囊底部剥离显露胆囊床"
}

# 动作组件翻译（用于正则组合 10-109 号标签）
COMPONENTS = {
    "grasper": "抓钳", "bipolar": "双极电凝", "hook": "电钩", 
    "scissors": "剪刀", "clipper": "夹闭器", "irrigator": "冲洗器",
    "dissect": "解剖", "grasp": "抓取", "retract": "牵引", 
    "coagulate": "电凝", "pack": "装袋", "clip": "夹闭", 
    "cut": "切断", "aspirate": "吸引", "irrigate": "冲洗",
    "null verb": "无动作", "cystic plate": "胆囊床", "gallbladder": "胆囊",
    "omentum": "网膜", "cystic artery": "胆囊动脉", "cystic duct": "胆囊管",
    "cystic pedicle": "胆囊蒂", "gut": "肠道", "liver": "肝脏",
    "peritoneum": "腹膜", "specimen bag": "标本袋", "abdominal wall cavity": "腹壁腔",
    "blood vessel": "血管", "adhesion": "粘连", "fluid": "液体", "null target": "无目标"
}

def translate_any_label(en_label):
    """
    智能翻译：先查全句表，查不到再按组件组合
    """
    en_label = en_label.strip()
    
    # 1. 尝试全句匹配 (解决 Phase 和 View)
    if en_label in FULL_LABEL_MAP:
        return FULL_LABEL_MAP[en_label]
    
    # 2. 尝试正则匹配组合动作 (针对 Action)
    match = re.search(r"tool (.*?) performing the action (.*?) on the target (.*)", en_label)
    if match:
        tool, action, target = match.groups()
        t_cn = COMPONENTS.get(tool.strip(), tool)
        a_cn = COMPONENTS.get(action.strip(), action)
        g_cn = COMPONENTS.get(target.strip(), target)
        return f"{t_cn}-{a_cn}-{g_cn}"
    
    # 3. 兜底处理
    return en_label.replace("of the phase ", "").replace("of ", "")

# ================= 3. 数据统计逻辑 (原封不动) =================
def count_train_labels():
    counts = np.zeros(NUM_CLASSES, dtype=int)
    mode = "train"
    pkl_p = os.path.join(DATA_PATH, f"cholec80/labels/{mode}/frame_phase_{mode}.pkl")
    if os.path.exists(pkl_p):
        invalid = [6, 10, 14, 32]
        with open(pkl_p, "rb") as f:
            data = pickle.load(f)
            for vid, frames in data.items():
                if int(vid[5:]) in invalid: continue
                for f in frames: counts[f["Phase_gt"]] += 1
    endo_p = os.path.join(DATA_PATH, "endoscapes/train/annotation_ds_coco.json")
    if os.path.exists(endo_p):
        with open(endo_p, 'r') as f:
            data = json.load(f)
            for img in data['images']:
                for i, val in enumerate(img['ds']):
                    if round(val) == 1: counts[i + 7] += 1
    vids = [1,2,4,5,13,15,18,22,23,25,26,27,31,35,36,40,43,47,48,49,52,56,57,60,62,65,66,68,70,75,79,92,96,103,110]
    c50_dir = os.path.join(DATA_PATH, "cholect50/labels")
    for v in vids:
        lp = os.path.join(c50_dir, f"VID{v:02d}.json" if v < 100 else f"VID{v:03d}.json")
        if os.path.exists(lp):
            with open(lp, 'r') as f:
                data = json.load(f)
                for _, gts in data['annotations'].items():
                    for ann in gts:
                        if ann[0] != -1: counts[ann[0] + 10] += 1
    return counts

# ================= 4. 绘图与字体应用 =================
def main():
    print("正在加载数据并翻译标签...")
    counts = count_train_labels()
    
    # 加载标签文件
    raw_en_labels = []
    if os.path.exists(LABEL_NAMES_FILE):
        with open(LABEL_NAMES_FILE, 'r') as f:
            raw_en_labels = [line.strip() for line in f.readlines() if line.strip()]
    
    # 生成横轴文本
    x_tick_labels = []
    for i in range(NUM_CLASSES):
        en_text = raw_en_labels[i] if i < len(raw_en_labels) else ""
        cn_text = translate_any_label(en_text)
        x_tick_labels.append(f"{i}: {cn_text}")

    # 创建画布 (增加高度容纳底部长名字)
    fig, ax = plt.subplots(figsize=(60, 25))
    x = np.arange(NUM_CLASSES)
    # 颜色：Phase(绿), View(橙), Action(蓝)
    colors = ['#27ae60']*7 + ['#e67e22']*3 + ['#2980b9']*(NUM_CLASSES-10)
    
    bars = ax.bar(x, counts, color=colors, edgecolor='black', linewidth=0.2)
    
    # 顶部数值标注 (应用本地字体)
    max_h = max(counts)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + (max_h * 0.005),
                f'{int(h)}', ha='center', va='bottom', rotation=90, 
                fontsize=15, fontweight='bold', fontproperties=font_prop)

    # 标题与坐标轴 (应用本地字体)
    ax.set_title('Cholec 系列数据集 - 训练集全量类别分布统计 (线性比例)', 
                 fontsize=48, pad=60, fontproperties=font_prop)
    ax.set_ylabel('图像样本张数', fontsize=36, fontproperties=font_prop)
    ax.set_xlabel('类别编号与名称', fontsize=36, fontproperties=font_prop)
    
    ax.set_xticks(x)
    # 横轴名字旋转 90 度并应用字体
    ax.set_xticklabels(x_tick_labels, rotation=90, fontsize=15, fontproperties=font_prop)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max_h * 1.15)

    # 图例设置 (应用本地字体)
    legend_elements = [
        Line2D([0], [0], color='#27ae60', lw=15, label='手术阶段 (Phase)'),
        Line2D([0], [0], color='#e67e22', lw=15, label='解剖关键视图 (View)'),
        Line2D([0], [0], color='#2980b9', lw=15, label='细粒度器械动作 (Action)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', prop={'fname': FONT_PATH, 'size': 32} if font_prop else None)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=120)
    print(f"\n生成成功！")
    print(f"1. 图片保存在: {os.path.abspath(SAVE_PATH)}")
    print(f"2. 如果中文还是显示不出来，请检查当前目录下是否有 {FONT_PATH} 文件。")

if __name__ == "__main__":
    main()