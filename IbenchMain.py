import argparse
from utils.config import Config
from ibench.evaluator.imageid import ImageidEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--config', default='/home/gdli7/IBench/config/imageid.py',
                        help='evaluation config file path')
    parser.add_argument('--data', default='/home/gdli7/IBench/data/chineseid_longer_v12_preview_0226.json', help='evaluation data file path')
    # unsplash50_short_v1.json
    # unsplash50_short_sdxl_instantid.json
    # unsplash50_short_sdxl_pulid.json
    # unsplash50_short_v11.json
    # unsplash50_short_v12_preview_bak.json
    # generateid_typemovie_v1.json
    # generateid_typemovie_v11.json
    # generateid_typemovie_v12_preview.json
    # chineseid_longer_v1.json
    # chineseid_longer_v11.json
    # chineseid_longer_v12_preview.json
    # generateid_typemovie_sdxl_instantid.json
    # generateid_typemovie_sdxl_pulid.json
    # chineseid_longer_sdxl_instantid.json
    # chineseid_longer_sdxl_pulid.json
    # chineseid_longer_v12_preview_0212.json
    # chineeseid_longer_v12_preview_0213.json
    # chineseid_longer_v12_preview_0217.json
    # chineseid_longer_v12_preview_0218_old.json
    # chineseid_longer_v12_preview_0219.json
    # chineseid_longer_v12_preview_0220.json
    # chineseid_longer_v12_preview_0221.json
    # chineseid_longer_v12_preview_0222.json
    # chineseid_longer_v12_preview_0223.json
    # chineseid_longer_v12_preview_0224.json
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
