import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.builder import METRICS
from utils.log import logger
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModel
from utils.compat import config
from utils.utils import LRUCache

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

DINO_MODEL_PATH = config.metrics.imageid.dinoeval.dino_model_path


class DINOModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DINOModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'device'):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # --------------------------------------------------------------------------------------------------------------
            logger.info(f"Loading dino model!!!")
            # import pdb;pdb.set_trace()
            self.model, self.processor = self.load_model()

    def load_model(self):
        # model = torch.hub.load(DINO_MODEL_PATH, 'dinov2_vitb14', source="local").to(self.device)
        processor = AutoImageProcessor.from_pretrained(DINO_MODEL_PATH)
        model = AutoModel.from_pretrained(DINO_MODEL_PATH)
        return model, processor


def dinov2_transform_Image(n_px):
    t = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return t


@METRICS.register_module()
class DINO(object):
    def __init__(self, type=""):
        # super().__init__()
        self.name = "dino"
        dino_model = DINOModel()
        self.model = dino_model.model
        self.processor = dino_model.processor
        self.device = dino_model.device
        self.embedding_cache = LRUCache(capacity=20)  # 设置缓存容量

    def get_embedding_with_cache(self, image_path):
        # 如果缓存中已有该图像路径对应的嵌入，则直接返回
        cached_embedding = self.embedding_cache.get(image_path)
        if cached_embedding is not None:
            return cached_embedding

        # 否则计算嵌入并存入缓存
        image = Image.open(image_path).convert('RGB')
        image = self.processor(images=image, return_tensors="pt")
        embedding = self.model(**image).last_hidden_state[:, 0].cpu()
        self.embedding_cache.put(image_path, embedding)
        return embedding

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        # image_transform = dinov2_transform_Image(224)

        dino_scores_list = []
        if category == "imageid":
            for data in tqdm(data_list):
                try:
                    id = data["id"]
                    # id = Image.open(id).convert('RGB')
                    # # id = np.array(id).astype(np.uint8)
                    # # id = image_transform(id)
                    # # id = id.unsqueeze(0).to(self.device)
                    # id = self.processor(images=id, return_tensors="pt")
                    # id_features = self.model(**id).last_hidden_state[:, 0].cpu()
                    # # id_features = F.normalize(id_features, dim=-1, p=2)
                    id_features = self.get_embedding_with_cache(id)

                    image = data["imagewithid"]
                    image = Image.open(image).convert('RGB')
                    # image = np.array(image).astype(np.uint8)
                    # image = image_transform(image)
                    # image = image.unsqueeze(0).to(self.device)
                    image = self.processor(images=image, return_tensors="pt")
                    image_features = self.model(**image).last_hidden_state[:, 0].cpu()
                    # image_features = F.normalize(image_features, dim=-1, p=2)

                    # similarity = F.cosine_similarity(id_features, image_features).item()
                    similarity = torch.nn.functional.cosine_similarity(id_features, image_features)
                    dino_scores_list.append(similarity.item())
                except Exception as e:
                    logger.error(f"处理项 {data} 出错：{e}")
                    continue
        cur_avg = np.mean(dino_scores_list)
        return cur_avg


"""
https://huggingface.co/blog/image-similarity"""
