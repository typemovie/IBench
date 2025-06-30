import os
import sys

import numpy as np

sys.path.append("/home/gdli7/IBench/ibench/metrics/t2i")

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from utils.builder import METRICS
from utils.log import logger
from utils.compat import config

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

LAION_AES_MODEL_PATH = config.metrics.t2i.aesthetic.laion_aes_model_path
LAION_MODEL_PATH = config.metrics.t2i.aesthetic.laion_model_path


def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC, antialias=False),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


@METRICS.register_module()
class AestheticScore(object):
    def __init__(self, batch_size=32, type=""):
        self.name = "laion_aesthetic_score"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        # ----------------------------------------------------------------------------------------------------------------
        logger.info(f"Loading {self.name} model!!!")
        self.aesthetic_model, self.clip_model, self.preprocess = self.load_model()

    def load_model(self):
        clip_model, preprocess = clip.load(LAION_MODEL_PATH, device='cuda', jit=False)
        clip_model.eval()

        aesthetic_model = nn.Linear(768, 1)
        aesthetic_model.load_state_dict(torch.load(LAION_AES_MODEL_PATH, weights_only=True))
        aesthetic_model.to(self.device)
        aesthetic_model.eval()
        return aesthetic_model, clip_model, preprocess

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        all_images = []
        # import pdb;pdb.set_trace()
        if category == "imageid":
            for data in data_list:
                images = data["imagewithid"]
                frame = Image.open(images)
                frame = frame.convert('RGB')
                frame = np.array(frame).astype(np.uint8)
                all_images.append(frame)
            buffer = np.array(all_images)
            frames = torch.Tensor(buffer)
            all_images = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

        image_transform = clip_transform(224)
        aesthetic_scores_list = []
        # import pdb;pdb.set_trace()
        for i in tqdm(range(0, len(all_images), self.batch_size)):
            image_batch = all_images[i:i + self.batch_size]
            image_batch = image_transform(image_batch)  # 4,3,224,,224
            image_batch = image_batch.to(self.device)

            with torch.no_grad():
                image_feats = self.clip_model.encode_image(image_batch).to(torch.float32)
                image_feats = F.normalize(image_feats, dim=-1, p=2)
                aesthetic_scores = self.aesthetic_model(image_feats).squeeze(dim=-1)
                aesthetic_scores_list.append(aesthetic_scores)

        aesthetic_scores = torch.cat(aesthetic_scores_list, dim=0)
        normalized_aesthetic_scores = aesthetic_scores / 10
        cur_avg = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True).item()

        return cur_avg
