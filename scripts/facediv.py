import cv2
import torch
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from utils.log import logger
from utils.builder import METRICS
from utils.compat import config
# from collections import OrderedDict
from utils.utils import LRUCache

FaceSim_MODEL_PATH = config.metrics.imageid.facesim.face_detection_model_path


def get_face_embedding(model, image_path):
    img = cv2.imread(image_path)
    faces = model.get(img)

    if len(faces) > 0:
        return faces[0].embedding
    return None


def cosine_similarity(a, b):
    a_normalized = a / np.linalg.norm(a)  # 归一化
    b_normalized = b / np.linalg.norm(b)
    return np.dot(a_normalized, b_normalized)


# @METRICS.register_module()
class FaceSim(object):
    def __init__(self, type=""):
        self.name = "facesim"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading {self.name} model!!!")
        self.model = self.load_model()
        self.embedding_cache = LRUCache(capacity=20)  # 设置缓存容量

    def load_model(self):
        model = FaceAnalysis(name="antelopev2", root=FaceSim_MODEL_PATH,
                             providers=['CUDAExecutionProvider'])
        model.prepare(ctx_id=1, det_size=(640, 640))
        return model

    def get_embedding_with_cache(self, image_path):
        # 如果缓存中已有该图像路径对应的嵌入，则直接返回
        cached_embedding = self.embedding_cache.get(image_path)
        if cached_embedding is not None:
            return cached_embedding

        # 否则计算嵌入并存入缓存
        embedding = get_face_embedding(self.model, image_path)
        self.embedding_cache.put(image_path, embedding)
        return embedding

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        facesim_scores_list = []
        if category == "imageid":
            for data in tqdm(data_list):
                try:
                    id = data['id']
                    id_embedding = self.get_embedding_with_cache(id)

                    images = data["imagewithid"]
                    imagewithid_embedding = get_face_embedding(self.model, images)
                    similarity = cosine_similarity(id_embedding, imagewithid_embedding)
                    facesim_scores_list.append(similarity)
                except:
                    print(data)
                    continue

        cur_avg = np.mean(facesim_scores_list)
        return cur_avg
