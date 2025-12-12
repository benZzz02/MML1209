import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import argparse
import re
import sys

# ==========================================
# 0. [核心] 字体配置逻辑
# ==========================================
def configure_font(font_path=None):
    """
    配置字体。
    1. 如果指定了 font_path，强制加载该文件。
    2. 如果没指定，尝试自动查找系统字体。
    """
    # --- 方案 A: 强制加载本地字体文件 (最稳妥) ---
    if font_path:
        if os.path.exists(font_path):
            try:
                # 将字体文件加入 Matplotlib 的管理器
                fm.fontManager.addfont(font_path)
                # 获取该字体的内部名称
                prop = fm.FontProperties(fname=font_path)
                custom_font_name = prop.get_name()
                
                # 设置为全局默认
                plt.rcParams['font.family'] = 'sans-serif' # 先设为无衬线
                plt.rcParams['font.sans-serif'] = [custom_font_name] # 首选该字体
                plt.rcParams['axes.unicode_minus'] = False # 修复负号显示
                
                print(f"✅ 已加载本地字体文件: {font_path}")
                print(f"   (字体注册名称: {custom_font_name})")
                return
            except Exception as e:
                print(f"❌ 加载本地字体失败: {e}")
                print("   -> 将尝试使用系统默认字体...")
        else:
            print(f"⚠️ 警告: 字体文件不存在 -> {font_path}")
            print("   -> 将尝试使用系统默认字体...")

    # --- 方案 B: 自动查找系统字体 (备选) ---
    print("🔄 正在扫描系统字体...")
    # 常用中文字体优先级
    candidates = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'PingFang SC', 'Heiti TC']
    system_fonts = {f.name for f in fm.fontManager.ttflist}
    
    found = False
    for name in candidates:
        if name in system_fonts:
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 使用系统字体: {name}")
            found = True
            break
            
    if not found:
        print("❌ 未找到中文字体！中文将无法显示。")
        print("💡 建议: 上传 SimHei.ttf 并使用 --font_file 参数。")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 防止报错

# ==========================================
# 1. 标签定义与翻译 (保持不变)
# ==========================================
FULL_RAW_PROMPTS = [
    "of the phase Preparation", "of the phase Calot Triangle Dissection", "of the phase Clipping Cutting", "of the phase Gallbladder Dissection", "of the phase Gallbladder Retraction", "of the phase Cleaning Coagulation", "of the phase Gallbladder Packaging",
    "of seeing two structures cystic duct and cystic artery", "of carefully dissected hepatocystic triangle presenting an unimpeded view of only the 2 cystic structures and the cystic plate", "of the lower part of the gallbladder divided from the liver bed to expose the cystic plate",
    "of the tool grasper performing the action dissect on the target cystic plate", "of the tool grasper performing the action dissect on the target gallbladder", "of the tool grasper performing the action dissect on the target omentum", "of the tool grasper performing the action grasp on the target cystic artery", "of the tool grasper performing the action grasp on the target cystic duct", "of the tool grasper performing the action grasp on the target cystic pedicle", "of the tool grasper performing the action grasp on the target cystic plate", "of the tool grasper performing the action grasp on the target gallbladder", "of the tool grasper performing the action grasp on the target gut", "of the tool grasper performing the action grasp on the target liver", "of the tool grasper performing the action grasp on the target omentum", "of the tool grasper performing the action grasp on the target peritoneum", "of the tool grasper performing the action grasp on the target specimen bag", "of the tool grasper performing the action pack on the target gallbladder", "of the tool grasper performing the action retract on the target cystic duct", "of the tool grasper performing the action retract on the target cystic pedicle", "of the tool grasper performing the action retract on the target cystic plate", "of the tool grasper performing the action retract on the target gallbladder", "of the tool grasper performing the action retract on the target gut", "of the tool grasper performing the action retract on the target liver", "of the tool grasper performing the action retract on the target omentum", "of the tool grasper performing the action retract on the target peritoneum",
    "of the tool bipolar performing the action coagulate on the target abdominal wall cavity", "of the tool bipolar performing the action coagulate on the target blood vessel", "of the tool bipolar performing the action coagulate on the target cystic artery", "of the tool bipolar performing the action coagulate on the target cystic duct", "of the tool bipolar performing the action coagulate on the target cystic pedicle", "of the tool bipolar performing the action coagulate on the target cystic plate", "of the tool bipolar performing the action coagulate on the target gallbladder", "of the tool bipolar performing the action coagulate on the target liver", "of the tool bipolar performing the action coagulate on the target omentum", "of the tool bipolar performing the action coagulate on the target peritoneum", "of the tool bipolar performing the action dissect on the target adhesion", "of the tool bipolar performing the action dissect on the target cystic artery", "of the tool bipolar performing the action dissect on the target cystic duct", "of the tool bipolar performing the action dissect on the target cystic plate", "of the tool bipolar performing the action dissect on the target gallbladder", "of the tool bipolar performing the action dissect on the target omentum", "of the tool bipolar performing the action grasp on the target cystic plate", "of the tool bipolar performing the action grasp on the target liver", "of the tool bipolar performing the action grasp on the target specimen bag", "of the tool bipolar performing the action retract on the target cystic duct", "of the tool bipolar performing the action retract on the target cystic pedicle", "of the tool bipolar performing the action retract on the target gallbladder", "of the tool bipolar performing the action retract on the target liver", "of the tool bipolar performing the action retract on the target omentum",
    "of the tool hook performing the action coagulate on the target blood vessel", "of the tool hook performing the action coagulate on the target cystic artery", "of the tool hook performing the action coagulate on the target cystic duct", "of the tool hook performing the action coagulate on the target cystic pedicle", "of the tool hook performing the action coagulate on the target cystic plate", "of the tool hook performing the action coagulate on the target gallbladder", "of the tool hook performing the action coagulate on the target liver", "of the tool hook performing the action coagulate on the target omentum", "of the tool hook performing the action cut on the target blood vessel", "of the tool hook performing the action cut on the target peritoneum", "of the tool hook performing the action dissect on the target blood vessel", "of the tool hook performing the action dissect on the target cystic artery", "of the tool hook performing the action dissect on the target cystic duct", "of the tool hook performing the action dissect on the target cystic plate", "of the tool hook performing the action dissect on the target gallbladder", "of the tool hook performing the action dissect on the target omentum", "of the tool hook performing the action dissect on the target peritoneum", "of the tool hook performing the action retract on the target gallbladder", "of the tool hook performing the action retract on the target liver",
    "of the tool scissors performing the action coagulate on the target omentum", "of the tool scissors performing the action cut on the target adhesion", "of the tool scissors performing the action cut on the target blood vessel", "of the tool scissors performing the action cut on the target cystic artery", "of the tool scissors performing the action cut on the target cystic duct", "of the tool scissors performing the action cut on the target cystic plate", "of the tool scissors performing the action cut on the target liver", "of the tool scissors performing the action cut on the target omentum", "of the tool scissors performing the action cut on the target peritoneum", "of the tool scissors performing the action dissect on the target cystic plate", "of the tool scissors performing the action dissect on the target gallbladder", "of the tool scissors performing the action dissect on the target omentum",
    "of the tool clipper performing the action clip on the target blood vessel", "of the tool clipper performing the action clip on the target cystic artery", "of the tool clipper performing the action clip on the target cystic duct", "of the tool clipper performing the action clip on the target cystic pedicle", "of the tool clipper performing the action clip on the target cystic plate",
    "of the tool irrigator performing the action aspirate on the target fluid", "of the tool irrigator performing the action dissect on the target cystic duct", "of the tool irrigator performing the action dissect on the target cystic pedicle", "of the tool irrigator performing the action dissect on the target cystic plate", "of the tool irrigator performing the action dissect on the target gallbladder", "of the tool irrigator performing the action dissect on the target omentum", "of the tool irrigator performing the action irrigate on the target abdominal wall cavity", "of the tool irrigator performing the action irrigate on the target cystic pedicle", "of the tool irrigator performing the action irrigate on the target liver", "of the tool irrigator performing the action retract on the target gallbladder", "of the tool irrigator performing the action retract on the target liver", "of the tool irrigator performing the action retract on the target omentum",
    "of the tool grasper performing the action null verb on the target null target", "of the tool bipolar performing the action null verb on the target null target", "of the tool hook performing the action null verb on the target null target", "of the tool scissors performing the action null verb on the target null target", "of the tool clipper performing the action null verb on the target null target", "of the tool irrigator performing the action null verb on the target null target"
]

CN_MAP = {
    'grasper': '抓钳', 'bipolar': '双极钳', 'hook': '电钩', 'scissors': '剪刀', 'clipper': '施夹器', 'irrigator': '冲吸器',
    'dissect': '解剖', 'grasp': '抓取', 'retract': '牵引', 'coagulate': '凝血', 'cut': '剪切', 'clip': '夹闭', 'aspirate': '吸取', 'irrigate': '冲洗', 'pack': '装袋', 'null verb': '无动作',
    'gallbladder': '胆囊', 'cystic plate': '胆囊板', 'omentum': '网膜', 'cystic artery': '胆囊动脉', 'cystic duct': '胆囊管', 'cystic pedicle': '胆囊蒂', 'gut': '肠道', 'liver': '肝脏', 'peritoneum': '腹膜', 'specimen bag': '标本袋', 'abdominal wall cavity': '腹腔壁', 'blood vessel': '血管', 'adhesion': '粘连', 'fluid': '液体', 'null target': '无目标'
}

def get_chinese_labels(raw_list):
    cn_list = []
    for item in raw_list:
        if "of the phase" in item:
            if "Preparation" in item: l = "准备阶段"
            elif "Calot Triangle Dissection" in item: l = "Calot三角解剖"
            elif "Clipping Cutting" in item: l = "夹闭剪断"
            elif "Gallbladder Dissection" in item: l = "胆囊解剖"
            elif "Gallbladder Retraction" in item: l = "胆囊牵引"
            elif "Cleaning Coagulation" in item: l = "清理凝血"
            elif "Gallbladder Packaging" in item: l = "胆囊装袋"
            else: l = "未知阶段"
            cn_list.append(l)
        elif "seeing two structures" in item: cn_list.append("视野:双结构")
        elif "hepatocystic triangle" in item: cn_list.append("视野:肝胆三角")
        elif "divided from the liver bed" in item: cn_list.append("视野:胆囊肝床")
        else:
            match = re.search(r"tool (.*?) performing the action (.*?) on the target (.*)", item)
            if match:
                t_en, a_en, tg_en = match.groups()
                t_cn = CN_MAP.get(t_en, t_en.capitalize())
                a_cn = CN_MAP.get(a_en, a_en.capitalize())
                tg_cn = CN_MAP.get(tg_en, tg_en.capitalize())
                cn_list.append(f"{t_cn}-{a_cn}-{tg_cn}")
            else:
                cn_list.append(item[:10])
    return cn_list

def _to_cn_type(eng_type):
    return {'Phase': '手术阶段', 'View': '安全视图', 'Action': '手术动作'}.get(eng_type, eng_type)

# ==========================================
# 2. 绘图与加载逻辑 (含 Gamma 校正)
# ==========================================
def plot_sub_matrix(matrix, row_names, col_names, title, save_path, gamma=2.0):
    if isinstance(matrix, torch.Tensor): matrix = matrix.cpu().numpy()
    
    # Gamma 校正
    matrix_enhanced = np.power(matrix, gamma)
    
    # 动态画布大小
    h_factor, w_factor = 0.6, 0.6
    h = min(max(len(row_names) * h_factor + 4, 8), 60)
    w = min(max(len(col_names) * w_factor + 4, 10), 60)

    plt.figure(figsize=(w, h))
    
    vmax = max(matrix_enhanced.max(), 0.01)

    ax = sns.heatmap(
        matrix_enhanced, 
        cmap='mako', 
        vmin=0.0, 
        vmax=vmax,
        square=True, 
        xticklabels=col_names, 
        yticklabels=row_names, 
        annot=False,
        cbar_kws={'label': f'相关性强度 (Gamma={gamma})', 'shrink': 0.5}
    )

    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    row_type, _, col_type = title.split(' ')
    cn_title = f"{_to_cn_type(row_type)} 与 {_to_cn_type(col_type)} 相关性矩阵"
    
    plt.title(cn_title, fontsize=20, pad=20, fontweight='bold')
    plt.xlabel(_to_cn_type(col_type), fontsize=16, fontweight='bold', labelpad=15)
    plt.ylabel(_to_cn_type(row_type), fontsize=16, fontweight='bold', labelpad=15)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ 保存: {save_path}")

def load_astar(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"❌ 错误: 文件不存在 {ckpt_path}"); sys.exit(1)
    try:
        print(f"📂 正在加载权重: {ckpt_path} ...")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        for k in state_dict.keys():
            if "A_star" in k: 
                print(f"✅ 成功找到矩阵 key: {k}")
                return state_dict[k]
        print("❌ 错误: 未在权重中找到 A_star。"); sys.exit(1)
    except Exception as e:
        print(f"❌ 加载异常: {e}"); sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help="权重文件路径")
    parser.add_argument('--out_dir', type=str, default="visual/sub_matrices_cn", help="输出文件夹")
    parser.add_argument('--gamma', type=float, default=2.0, help="Gamma校正系数")
    # [关键] 直接指定字体文件
    parser.add_argument('--font_file', type=str, default=None, help="本地中文字体文件路径 (如 SimHei.ttf)")
    
    args = parser.parse_args()

    # 1. 配置字体
    configure_font(args.font_file)

    # 2. 准备数据
    cn_names = get_chinese_labels(FULL_RAW_PROMPTS)
    full_matrix = load_astar(args.ckpt)

    if full_matrix is not None:
        indices = {'Phase': (0, 7), 'View': (7, 10), 'Action': (10, 110)}
        combinations = [
            ('Phase', 'Action'), ('Phase', 'Phase'), ('View', 'Action'),
            ('View', 'View'), ('Action', 'Action'), ('Phase', 'View')
        ]

        print(f"🚀 开始生成中文图表...")
        for row_key, col_key in combinations:
            r_start, r_end = indices[row_key]
            c_start, c_end = indices[col_key]
            
            sub_mat = full_matrix[r_start:r_end, c_start:c_end]
            sub_row_names = cn_names[r_start:r_end]
            sub_col_names = cn_names[c_start:c_end]
            
            fname = f"{row_key}_vs_{col_key}_cn.png"
            plot_sub_matrix(sub_mat, sub_row_names, sub_col_names, 
                            title=f"{row_key} vs {col_key}", 
                            save_path=os.path.join(args.out_dir, fname), 
                            gamma=args.gamma)
            
        print(f"\n🎉 全部完成！结果保存在: {args.out_dir}")