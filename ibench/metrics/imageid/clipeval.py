import clip
import numpy as np
import torch

from tqdm import tqdm
from PIL import Image
from utils.builder import METRICS
from torchvision import transforms
from torchvision.transforms import ToTensor
from utils.log import logger
from utils.compat import config
from utils.utils import LRUCache

CLIP_MODEL_PATH = config.metrics.imageid.clipeval.clip_model_path


class ClipModel:
    # 单例模式
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ClipModel, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'device'):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # -------------------------------------------------------------------------------------------------------------------
            logger.info("Loading clip model!!!!")
            self.model, self.clip_preprocess = self.load_model()

            self.preprocess = transforms.Compose([transforms.Normalize(
                mean=[-1.0, -1.0, -1.0],
                std=[2.0, 2.0, 2.0])] +  # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                                 # to match CLIP input scale assumptions
                                                 self.clip_preprocess.transforms[:2] + self.clip_preprocess.transforms[
                                                                                       4:])

    def load_model(self):
        model, clip_preprocess = clip.load(CLIP_MODEL_PATH, device=self.device, jit=False)
        return model, clip_preprocess


class ClipEval(object):
    def __init__(self):
        clip_model = ClipModel()
        self.device = clip_model.device
        self.model = clip_model.model
        self.preprocess = clip_model.preprocess

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True, truncate: bool = True) -> torch.Tensor:
        tokens = clip.tokenize(text, truncate=True).to(self.device)
        text_features = self.encode_text(tokens).detach()
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def img_to_img_similarity(self, imagewithid_tensor,
                              imagewithoutid_tensor):  # imagewithid_tensor, imagewithoutid_tensor
        imagewithid_img_features = self.get_image_features(imagewithid_tensor)
        imagewithoutid_img_features = self.get_image_features(imagewithoutid_tensor)

        # return (src_img_features @ gen_img_features.T).mean()
        return torch.bmm(imagewithid_img_features.unsqueeze(1), imagewithoutid_img_features.unsqueeze(2)).mean()

    def txt_to_img_similarity(self, text, generated_images, truncate):
        text_features = self.get_text_features(text, truncate)
        gen_img_features = self.get_image_features(generated_images)

        if text_features.shape[0] != gen_img_features.shape[0]:
            text_features = text_features.repeat(gen_img_features.shape[0], 1)

        # return (text_features @ gen_img_features.T).mean()
        return torch.bmm(text_features.unsqueeze(1), gen_img_features.unsqueeze(2)).mean()


@METRICS.register_module()
class ClipI(ClipEval):
    def __init__(self, type=""):
        super().__init__()
        self.name = "clip-i"
        self.embedding_cache = LRUCache(capacity=20)  # 设置缓存容量

    def get_image_features_with_cache(self, image_path):
        # 如果缓存中已有该图像路径对应的嵌入，则直接返回
        cached_embedding = self.embedding_cache.get(image_path)
        if cached_embedding is not None:
            return cached_embedding

        # 否则计算嵌入并存入缓存
        image = Image.open(image_path).convert('RGB')
        image_tensor = ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
        embedding = self.get_image_features(image_tensor)
        self.embedding_cache.put(image_path, embedding)
        return embedding

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        clipi_scores_list = []
        if category == "imageid":
            for data in tqdm(data_list):
                imagewithid = data['imagewithid']
                imagewithoutid = data['imagewithoutid']

                # imagewithid = Image.open(imagewithid).convert('RGB')
                # imagewithid_tensor = (
                #         ToTensor()(imagewithid).unsqueeze(0) * 2.0 - 1.0)
                imagewithid_img_features = self.get_image_features_with_cache(imagewithid)

                imagewithoutid = Image.open(imagewithoutid).convert("RGB")
                imagewithoutid_tensor = (
                        ToTensor()(imagewithoutid).unsqueeze(0) * 2.0 - 1.0)
                imagewithoutid_img_features = self.get_image_features(imagewithoutid_tensor)

                similarity = torch.bmm(imagewithid_img_features.unsqueeze(1),
                                       imagewithoutid_img_features.unsqueeze(2)).mean()
                # similarity = self.img_to_img_similarity(imagewithid_tensor, imagewithoutid_tensor)

                clipi_scores_list.append(similarity.item())
        cur_avg = np.mean(clipi_scores_list)
        return cur_avg


@METRICS.register_module()
class ClipT(ClipEval):
    def __init__(self, truncate=True, type=""):
        super().__init__()
        self.name = "clip-t"

        self.truncate = truncate

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        clipt_scores_list = []
        if category == "imageid":
            for data in tqdm(data_list):
                prompt = data['prompt']
                # imagewithoutid = data['imagewithoutid']
                imagewithid = data['imagewithid']
                imagewithid = Image.open(imagewithid).convert("RGB")
                imagewithid_tensor = (
                        ToTensor()(imagewithid).unsqueeze(0) * 2.0 - 1.0)
                similarity = self.txt_to_img_similarity(prompt, imagewithid_tensor, self.truncate)
                clipt_scores_list.append(similarity.item())
        cur_avg = np.mean(clipt_scores_list)
        return cur_avg


"""
https://github.com/mit-han-lab/fastcomposer/blob/main/evaluation/single_object/single_object_evaluation.py#L118
"""
