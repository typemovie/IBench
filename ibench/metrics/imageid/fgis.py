"""
consistentid
生成图片与参考图像中生成面部区域的dino特征融合之间的平均余弦相似度
"""
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.builder import METRICS
from utils.utils import LRUCache
from .posediv import MTCNNDet
from .dinoeval import DINOModel
from utils.log import logger

@METRICS.register_module()
class FGIS:
    def __init__(self, type=""):
        self.name = "fgis"

        dino_model = DINOModel()
        self.model = dino_model.model
        self.processor = dino_model.processor

        mtcnn_model = MTCNNDet()
        self.face_detector = mtcnn_model.mtcnn_model
        self.embedding_cache = LRUCache(capacity=20)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def face_detect(self, img):
        img = cv2.imread(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.face_detector.detect_faces(img_rgb)
        if len(detections) > 0:
            for detection in detections:
                x, y, w, h = detection["box"]
                return [x, y, w, h]
        else:
            return None

    def get_embedding_with_cache(self, image_path):
        # 如果缓存中已有该图像路径对应的嵌入，则直接返回
        cached_embedding = self.embedding_cache.get(image_path)
        if cached_embedding is not None:
            return cached_embedding

        # 否则计算嵌入并存入缓存
        image = Image.open(image_path).convert('RGB')
        region = self.face_detect(image_path)
        if region is not None:
            x, y, w, h = region
            image_face = image.crop((int(x - 30), int(y - 30), int(x + w + 30), int(y + h + 30)))
        else:
            image_face = image

        image_face = self.processor(images=image_face, return_tensors="pt")
        embedding = self.model(**image_face).last_hidden_state[:, 0].cpu()
        self.embedding_cache.put(image_path, embedding)
        return embedding

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        fgis_scores_list = []
        if category == "imageid":
            for data in tqdm(data_list):
                try:
                    id = data['id']
                    # region = self.face_detect(id)
                    # id = Image.open(id).convert('RGB')
                    # if region is not None:
                    #     x, y, w, h = region
                    #     id_face = id.crop((int(x - 30), int(y - 30), int(x + w + 30), int(y + h + 30)))
                    # else:
                    #     id_face = id
                    # id_face = self.processor(images=id_face, return_tensors="pt")
                    # id_features = self.model(**id_face).last_hidden_state[:, 0].cpu()
                    id_features = self.get_embedding_with_cache(id)

                    imagewithid = data['imagewithid']
                    region = self.face_detect(imagewithid)
                    imagewithid = Image.open(imagewithid).convert('RGB')
                    if region is not None:
                        x, y, w, h = region
                        imagewithid_face = imagewithid.crop(
                            (int(x - 30), int(y - 30), int(x + w + 30), int(y + h + 30)))
                    else:
                        imagewithid_face = imagewithid
                    imagewithid_face = self.processor(images=imagewithid_face, return_tensors="pt")
                    image_features = self.model(**imagewithid_face).last_hidden_state[:, 0].cpu()

                    similarity = torch.nn.functional.cosine_similarity(id_features, image_features)
                    fgis_scores_list.append(similarity.item())
                except Exception as e:
                    logger.error(f"处理项 {data} 出错：{e}")
                    continue
        cur_avg = np.mean(fgis_scores_list)
        return cur_avg
