import argparse
from utils.config import Config
from ibench.evaluator.imageid import ImageidEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--config', default='/home/gdli7/IBench/config/imageid.py',
                        help='evaluation config file path')
    parser.add_argument('--data', default='/home/gdli7/IBench/data/chineseid_longer_v12_preview_0226.json', help='evaluation data file path')
    args = parser.parse_args()
    parser.add_argument('--category', default='imageid')
    parser.add_argument('--output', default='', help='evaluation output file path')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    config = Config.fromfile(args.config)
    if args.category == "imageid":
        result = ImageidEvaluator(config, args.data).evaluate()
