import numpy as np
import torch

from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
from dreamsim import dreamsim
from utils.builder import METRICS
from utils.log import logger
from utils.compat import config
from utils.utils import LRUCache
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

DREAMSIM_MODEL_PATH = config.metrics.imageid.dreamsimeval.dreamsim_model_path


def dreamsim_transform_Image(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=BICUBIC),
        ToTensor(),
    ])


@METRICS.register_module()
class DreamSim(object):
    def __init__(self, type=""):
        self.name = "dreamsim"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ----------------------------------------------------------
        logger.info(f"Loading {self.name} model!!!!")
        self.model = self.load_model()
        self.embedding_cache = LRUCache(capacity=20)  # 设置缓存容量

    def load_model(self):
        model, _ = dreamsim(pretrained=True, cache_dir=DREAMSIM_MODEL_PATH)
        return model

    def get_embedding_with_cache(self, image_path):
        # 如果缓存中已有该图像路径对应的嵌入，则直接返回
        cached_embedding = self.embedding_cache.get(image_path)
        if cached_embedding is not None:
            return cached_embedding

        # 否则计算嵌入并存入缓存
        image_transform = dreamsim_transform_Image(224)
        image = image_transform(Image.open(image_path).convert('RGB'))
        image = image.unsqueeze(0).to(self.device)
        embedding = self.model(image)
        embedding = F.normalize(embedding, dim=-1, p=2)
        self.embedding_cache.put(image_path, embedding)
        return embedding

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        dreamsim_scores_list = []
        if category == "imageid":
            image_transform = dreamsim_transform_Image(224)
            for data in tqdm(data_list):
                id = data['id']
                # id = image_transform(Image.open(id).convert('RGB'))
                # id = id.unsqueeze(0).to(self.device)
                # id_feature = self.model(id)
                # id_feature = F.normalize(id_feature, dim=-1, p=2)
                id_feature = self.get_embedding_with_cache(id)

                image = data['imagewithid']
                image = image_transform(Image.open(image).convert('RGB'))
                image = image.unsqueeze(0).to(self.device)
                image_feature = self.model(image)
                image_feature = F.normalize(image_feature, dim=-1, p=2)

                similarity = F.cosine_similarity(id_feature, image_feature).item()
                dreamsim_scores_list.append(similarity)
        cur_avg = np.mean(dreamsim_scores_list)
        return cur_avg
