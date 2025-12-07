import argparse

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
args = parser.parse_args()
