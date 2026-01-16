import argparse
import os

parser = argparse.ArgumentParser(description='Cholec Data Training')
parser.add_argument('-c',
                    '--config-file',
                    help='config file',
                    default='configs/surgadapt+cholec.yaml',
                    type=str)
parser.add_argument('-t',
                    '--test',
                    help='run test',
                    default=False,
                    action="store_true")
parser.add_argument('-r', '--round', help='round', default=1, type=int)
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument("--weights", default=None, type=str, help="Path to the specific checkpoint to test")

# ------------------- SwanLab CLI params (新增，含默认值) -------------------
parser.add_argument("--swan_project", type=str, default="SCPNet", help="SwanLab project name")

# 默认 group：取 config 文件名（不含后缀），避免所有实验挤在一起
default_group = os.path.splitext(os.path.basename('configs/surgadapt+cholec.yaml'))[0]
parser.add_argument("--exp_group", type=str, default=default_group,
                    help="SwanLab group name. Default: derived from --config-file basename")

# 默认 exp_name：None -> 训练脚本里自动生成
parser.add_argument("--exp_name", type=str, default=None,
                    help="SwanLab experiment name (run name). Default: auto-generated in script.")

# 默认记录的配置文件：config.py + args.py + 你指定的 yaml 配置
parser.add_argument("--log_config_files", type=str, nargs="*", default=None,
                    help="Config files to log as text to SwanLab. Default: [config.py, args.py, <config-file>].")
# ------------------------------------------------------------------------

args = parser.parse_args()

# 把默认 group 动态改成用户传的 config-file 名（真正生效的默认）
if args.exp_group == default_group:
    args.exp_group = os.path.splitext(os.path.basename(args.config_file))[0]

# log_config_files 不传时，用默认列表（包含 yaml 配置）
if args.log_config_files is None:
    args.log_config_files = ["config.py", "args.py", args.config_file]
