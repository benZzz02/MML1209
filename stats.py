import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import re

# ================= 1. 配置与字体加载 =================
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

# 只生成排序后的图
SAVE_PATH_SORTED = "train_dist_final_chinese_sorted.png"

# ================= 2. 翻译映射 =================
FULL_LABEL_MAP = {
    # Phase
    "of the phase Preparation": "准备阶段",
    "of the phase Calot Triangle Dissection": "胆囊三角解剖",
    "of the phase Clipping Cutting": "管路夹闭切断",
    "of the phase Gallbladder Dissection": "胆囊壁剥离",
    "of the phase Gallbladder Retraction": "胆囊提取",
    "of the phase Cleaning Coagulation": "冲洗与止血",
    "of the phase Gallbladder Packaging": "胆囊装袋",

    # View
    "of seeing two structures cystic duct and cystic artery": "观察到胆囊管与胆囊动脉",
    "of carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate": "CVS关键安全视野(胆囊三角解剖清晰)",
    "of the lower part of the gallbladder divided from the liver bed to expose the cystic plate": "胆囊底部剥离显露胆囊床"
}

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

def translate_any_label(en_label: str) -> str:
    en_label = (en_label or "").strip()

    if en_label in FULL_LABEL_MAP:
        return FULL_LABEL_MAP[en_label]

    match = re.search(r"tool (.*?) performing the action (.*?) on the target (.*)", en_label)
    if match:
        tool, action, target = match.groups()
        t_cn = COMPONENTS.get(tool.strip(), tool.strip())
        a_cn = COMPONENTS.get(action.strip(), action.strip())
        g_cn = COMPONENTS.get(target.strip(), target.strip())
        return f"{t_cn}-{a_cn}-{g_cn}"

    return en_label.replace("of the phase ", "").replace("of ", "")

# ================= 3. 数据统计逻辑 (原封不动) =================
def count_train_labels() -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=int)
    mode = "train"

    pkl_p = os.path.join(DATA_PATH, f"cholec80/labels/{mode}/frame_phase_{mode}.pkl")
    if os.path.exists(pkl_p):
        invalid = [6, 10, 14, 32]
        with open(pkl_p, "rb") as f:
            data = pickle.load(f)
            for vid, frames in data.items():
                if int(vid[5:]) in invalid:
                    continue
                for fr in frames:
                    counts[fr["Phase_gt"]] += 1

    endo_p = os.path.join(DATA_PATH, "endoscapes/train/annotation_ds_coco.json")
    if os.path.exists(endo_p):
        with open(endo_p, 'r') as f:
            data = json.load(f)
            for img in data['images']:
                for i, val in enumerate(img['ds']):
                    if round(val) == 1:
                        counts[i + 7] += 1

    vids = [1,2,4,5,13,15,18,22,23,25,26,27,31,35,36,40,43,47,48,49,52,56,57,60,62,65,66,68,70,75,79,92,96,103,110]
    c50_dir = os.path.join(DATA_PATH, "cholect50/labels")
    for v in vids:
        lp = os.path.join(c50_dir, f"VID{v:02d}.json" if v < 100 else f"VID{v:03d}.json")
        if os.path.exists(lp):
            with open(lp, 'r') as f:
                data = json.load(f)
                for _, gts in data['annotations'].items():
                    for ann in gts:
                        if ann[0] != -1:
                            counts[ann[0] + 10] += 1

    return counts

# ================= 4. 绘图（排序版） =================
def main():
    print("正在加载数据并翻译标签...")
    counts = count_train_labels()

    # 读英文标签
    raw_en_labels = []
    if os.path.exists(LABEL_NAMES_FILE):
        with open(LABEL_NAMES_FILE, 'r') as f:
            raw_en_labels = [line.strip() for line in f.readlines() if line.strip()]
    else:
        print(f"警告: 未找到标签文件 {LABEL_NAMES_FILE}，横轴标签将为空或不完整。")

    # 每个 class 的中文名（按原 class id）
    class_cn = []
    for i in range(NUM_CLASSES):
        en_text = raw_en_labels[i] if i < len(raw_en_labels) else ""
        class_cn.append(translate_any_label(en_text))

    # 原 class 的颜色：Phase(绿), View(橙), Action(蓝)
    class_colors = ['#27ae60'] * 7 + ['#e67e22'] * 3 + ['#2980b9'] * (NUM_CLASSES - 10)

    # ================= 关键：从大到小排序（稳定） =================
    # 先按 counts 降序，再按 class_id 升序（同count时固定顺序）
    class_ids = np.arange(NUM_CLASSES)
    sort_idx = np.lexsort((class_ids, -counts))  # stable deterministic

    counts_sorted = counts[sort_idx]
    colors_sorted = [class_colors[i] for i in sort_idx]

    # x轴按排名显示（0..109）
    x = np.arange(NUM_CLASSES)

    # 显示为 Rank + 原始ID，避免你误以为“没排序”
    x_tick_labels_sorted = [
        f"Rank {rank:03d} | ID {cid:03d}: {class_cn[cid]}"
        for rank, cid in enumerate(sort_idx)
    ]

    # ================= 开始画图 =================
    fig, ax = plt.subplots(figsize=(60, 25))
    bars = ax.bar(x, counts_sorted, color=colors_sorted, edgecolor='black', linewidth=0.2)

    max_h = int(np.max(counts_sorted)) if len(counts_sorted) else 0
    for bar, h in zip(bars, counts_sorted):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + (max_h * 0.005),
            f"{int(h)}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=15,
            fontweight="bold",
            fontproperties=font_prop
        )

    ax.set_title('Cholec 系列数据集 - 训练集全量类别分布统计 (线性比例, 从大到小排序)',
                 fontsize=48, pad=60, fontproperties=font_prop)
    ax.set_ylabel('图像样本张数', fontsize=36, fontproperties=font_prop)
    ax.set_xlabel('按样本数排序后的排名 (Rank) 与 原始类别ID', fontsize=36, fontproperties=font_prop)

    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels_sorted, rotation=90, fontsize=15, fontproperties=font_prop)

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max_h * 1.15 if max_h > 0 else 1)

    legend_elements = [
        Line2D([0], [0], color='#27ae60', lw=15, label='手术阶段 (Phase)'),
        Line2D([0], [0], color='#e67e22', lw=15, label='解剖关键视图 (View)'),
        Line2D([0], [0], color='#2980b9', lw=15, label='细粒度器械动作 (Action)')
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              prop={'fname': FONT_PATH, 'size': 32} if font_prop else None)

    plt.tight_layout()
    plt.savefig(SAVE_PATH_SORTED, dpi=120)
    plt.close(fig)

    print("\n生成成功！")
    print(f"排序图保存在: {os.path.abspath(SAVE_PATH_SORTED)}")
    print("检查方法：左边第一根柱子一定是全局最大计数，右边是最小/0 计数。")
    if not os.path.exists(FONT_PATH):
        print(f"提示：如果中文显示不出来，请确认当前目录下是否有字体文件 {FONT_PATH}。")

if __name__ == "__main__":
    main()
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
import re

# ================= 1. 配置与字体加载 =================
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

# 只生成排序后的图
SAVE_PATH_SORTED = "train_dist_final_chinese_sorted.png"

# ================= 2. 翻译映射 =================
FULL_LABEL_MAP = {
    # Phase
    "of the phase Preparation": "准备阶段",
    "of the phase Calot Triangle Dissection": "胆囊三角解剖",
    "of the phase Clipping Cutting": "管路夹闭切断",
    "of the phase Gallbladder Dissection": "胆囊壁剥离",
    "of the phase Gallbladder Retraction": "胆囊提取",
    "of the phase Cleaning Coagulation": "冲洗与止血",
    "of the phase Gallbladder Packaging": "胆囊装袋",

    # View
    "of seeing two structures cystic duct and cystic artery": "观察到胆囊管与胆囊动脉",
    "of carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate": "CVS关键安全视野(胆囊三角解剖清晰)",
    "of the lower part of the gallbladder divided from the liver bed to expose the cystic plate": "胆囊底部剥离显露胆囊床"
}

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

def translate_any_label(en_label: str) -> str:
    en_label = (en_label or "").strip()

    if en_label in FULL_LABEL_MAP:
        return FULL_LABEL_MAP[en_label]

    match = re.search(r"tool (.*?) performing the action (.*?) on the target (.*)", en_label)
    if match:
        tool, action, target = match.groups()
        t_cn = COMPONENTS.get(tool.strip(), tool.strip())
        a_cn = COMPONENTS.get(action.strip(), action.strip())
        g_cn = COMPONENTS.get(target.strip(), target.strip())
        return f"{t_cn}-{a_cn}-{g_cn}"

    return en_label.replace("of the phase ", "").replace("of ", "")

# ================= 3. 数据统计逻辑 (原封不动) =================
def count_train_labels() -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=int)
    mode = "train"

    pkl_p = os.path.join(DATA_PATH, f"cholec80/labels/{mode}/frame_phase_{mode}.pkl")
    if os.path.exists(pkl_p):
        invalid = [6, 10, 14, 32]
        with open(pkl_p, "rb") as f:
            data = pickle.load(f)
            for vid, frames in data.items():
                if int(vid[5:]) in invalid:
                    continue
                for fr in frames:
                    counts[fr["Phase_gt"]] += 1

    endo_p = os.path.join(DATA_PATH, "endoscapes/train/annotation_ds_coco.json")
    if os.path.exists(endo_p):
        with open(endo_p, 'r') as f:
            data = json.load(f)
            for img in data['images']:
                for i, val in enumerate(img['ds']):
                    if round(val) == 1:
                        counts[i + 7] += 1

    vids = [1,2,4,5,13,15,18,22,23,25,26,27,31,35,36,40,43,47,48,49,52,56,57,60,62,65,66,68,70,75,79,92,96,103,110]
    c50_dir = os.path.join(DATA_PATH, "cholect50/labels")
    for v in vids:
        lp = os.path.join(c50_dir, f"VID{v:02d}.json" if v < 100 else f"VID{v:03d}.json")
        if os.path.exists(lp):
            with open(lp, 'r') as f:
                data = json.load(f)
                for _, gts in data['annotations'].items():
                    for ann in gts:
                        if ann[0] != -1:
                            counts[ann[0] + 10] += 1

    return counts

# ================= 4. 绘图（排序版） =================
def main():
    print("正在加载数据并翻译标签...")
    counts = count_train_labels()

    # 读英文标签
    raw_en_labels = []
    if os.path.exists(LABEL_NAMES_FILE):
        with open(LABEL_NAMES_FILE, 'r') as f:
            raw_en_labels = [line.strip() for line in f.readlines() if line.strip()]
    else:
        print(f"警告: 未找到标签文件 {LABEL_NAMES_FILE}，横轴标签将为空或不完整。")

    # 每个 class 的中文名（按原 class id）
    class_cn = []
    for i in range(NUM_CLASSES):
        en_text = raw_en_labels[i] if i < len(raw_en_labels) else ""
        class_cn.append(translate_any_label(en_text))

    # 原 class 的颜色：Phase(绿), View(橙), Action(蓝)
    class_colors = ['#27ae60'] * 7 + ['#e67e22'] * 3 + ['#2980b9'] * (NUM_CLASSES - 10)

    # ================= 关键：从大到小排序（稳定） =================
    # 先按 counts 降序，再按 class_id 升序（同count时固定顺序）
    class_ids = np.arange(NUM_CLASSES)
    sort_idx = np.lexsort((class_ids, -counts))  # stable deterministic

    counts_sorted = counts[sort_idx]
    colors_sorted = [class_colors[i] for i in sort_idx]

    # x轴按排名显示（0..109）
    x = np.arange(NUM_CLASSES)

    # 显示为 Rank + 原始ID，避免你误以为“没排序”
    x_tick_labels_sorted = [
        f"Rank {rank:03d} | ID {cid:03d}: {class_cn[cid]}"
        for rank, cid in enumerate(sort_idx)
    ]

    # ================= 开始画图 =================
    fig, ax = plt.subplots(figsize=(60, 25))
    bars = ax.bar(x, counts_sorted, color=colors_sorted, edgecolor='black', linewidth=0.2)

    max_h = int(np.max(counts_sorted)) if len(counts_sorted) else 0
    for bar, h in zip(bars, counts_sorted):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + (max_h * 0.005),
            f"{int(h)}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=15,
            fontweight="bold",
            fontproperties=font_prop
        )

    ax.set_title('Cholec 系列数据集 - 训练集全量类别分布统计 (线性比例, 从大到小排序)',
                 fontsize=48, pad=60, fontproperties=font_prop)
    ax.set_ylabel('图像样本张数', fontsize=36, fontproperties=font_prop)
    ax.set_xlabel('按样本数排序后的排名 (Rank) 与 原始类别ID', fontsize=36, fontproperties=font_prop)

    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels_sorted, rotation=90, fontsize=15, fontproperties=font_prop)

    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max_h * 1.15 if max_h > 0 else 1)

    legend_elements = [
        Line2D([0], [0], color='#27ae60', lw=15, label='手术阶段 (Phase)'),
        Line2D([0], [0], color='#e67e22', lw=15, label='解剖关键视图 (View)'),
        Line2D([0], [0], color='#2980b9', lw=15, label='细粒度器械动作 (Action)')
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              prop={'fname': FONT_PATH, 'size': 32} if font_prop else None)

    plt.tight_layout()
    plt.savefig(SAVE_PATH_SORTED, dpi=120)
    plt.close(fig)

    print("\n生成成功！")
    print(f"排序图保存在: {os.path.abspath(SAVE_PATH_SORTED)}")
    print("检查方法：左边第一根柱子一定是全局最大计数，右边是最小/0 计数。")
    if not os.path.exists(FONT_PATH):
        print(f"提示：如果中文显示不出来，请确认当前目录下是否有字体文件 {FONT_PATH}。")

if __name__ == "__main__":
    main()
