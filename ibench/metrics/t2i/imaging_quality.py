import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ

from pathlib import Path
from omegaconf import OmegaConf
from utils.log import logger
from utils.builder import METRICS
from utils.compat import config

MUSIQ_MODEL_PATH = config.metrics.t2i.imaging_quality.musiq_model_path


def transform(images, preprocess_mode='shorter'):
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h, w) > 512:
            scale = 512. / min(h, w)
            images = transforms.Resize(size=(int(scale * h), int(scale * w)), antialias=False)(images)
            if preprocess_mode == 'shorter_centercrop':
                images = transforms.CenterCrop(512)(images)
    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h, w) > 512:
            scale = 512. / max(h, w)
            images = transforms.Resize(size=(int(scale * h), int(scale * w)), antialias=False)(images)
    elif preprocess_mode == 'None':
        return images / 255.
    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.


@METRICS.register_module()
class ImagingQuality(object):
    def __init__(self, batch_size, type=""):
        self.name = "imagequality"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # import pdb;pdb.set_trace()
        self.batch_size = batch_size
        # ---------------------------------------------------------------------
        logger.info(f"Loading {self.name} model!!!")
        self.model = self.load_model()

    def load_model(self):
        model = MUSIQ(pretrained_model_path=MUSIQ_MODEL_PATH)
        model.to(self.device)
        model.training = False
        return model

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        all_images = []
        if category == "imageid":
            for data in tqdm(data_list):
                images = data["imagewithid"]
                frame = Image.open(images)
                frame = frame.convert('RGB')
                frame = np.array(frame).astype(np.uint8)
                all_images.append(frame)
            buffer = np.array(all_images)
            frames = torch.Tensor(buffer)
            all_images = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

        images = transform(all_images, preprocess_mode='longer')
        acc_score_list = []
        # import pdb; pdb.set_trace()
        # for i in range(len(images)):
        #     frame = images[i].unsqueeze(0).to(self.device)
        for i in range(0, len(all_images), self.batch_size):
            image_batch = all_images[i:i + self.batch_size]
            image_batch = image_batch.to(self.device)
            with torch.no_grad():
                score = self.model(image_batch).squeeze(dim=-1)
                acc_score_list.append(score)

        acc_score = torch.cat(acc_score_list, dim=0)
        normalized_acc_scores = acc_score / 100
        cur_avg = torch.mean(normalized_acc_scores, dim=0, keepdim=True).item()
        return cur_avg
