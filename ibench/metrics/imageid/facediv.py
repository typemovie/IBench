import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from torchvision import transforms
from mtcnn import MTCNN
from math import exp

# from insightface.app import FaceAnalysis
from utils.log import logger
from utils.builder import METRICS
from utils.compat import config
from utils.utils import LRUCache
from .facesim import FaceModelLoader

# 配置相关路径
FaceSim_MODEL_PATH = config.metrics.imageid.facesim.face_detection_model_path


def get_face_embedding(model, image_path):
    """
    加载图像并使用人脸模型计算嵌入
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = model.get(img)
    if len(faces) > 0:
        return faces[0].embedding
    return None


def cosine_similarity(a, b):
    a_normalized = a / np.linalg.norm(a)
    b_normalized = b / np.linalg.norm(b)
    return np.dot(a_normalized, b_normalized)


@METRICS.register_module()
class FaceDiv1(object):
    """
    FaceDivNormalized 指标：在衡量 id 与 imagewithid 之间的人脸相似性的同时，
    采用归一化后的人脸关键点差异衡量编辑性变化。

    原始方法直接计算关键点的欧式距离，可能数值较大且无法反映编辑性差异，
    这里我们采用如下优化方法：
      - 对于每一幅图像，从关键点计算出中心和尺度（如使用关键点的最小外接矩形对角线长度）；
      - 将关键点坐标归一化，即 (x - center_x) / scale, (y - center_y) / scale；
      - 计算归一化后的欧式距离差异，从而获得一个无量纲、稳定的数值，再取平均作为 landmark_diff。
    """

    def __init__(self, type=""):
        self.name = "facediv"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载用于计算嵌入的人脸模型
        self.face_model = FaceModelLoader().get_model()
        self.embedding_cache = LRUCache(capacity=20)

        # 初始化 MTCNN 检测器用于提取人脸关键点
        self.face_detector = MTCNN()
        self.landmark_cache = LRUCache(capacity=20)
        self.keys = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        # self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # 设置理想关键点变化目标 T 和标准差 sigma（用于后续调控）
        self.T = 0.2  # 归一化后理想的形状差异（无量纲）
        self.sigma = 0.1  # 控制调控函数的宽度

    def get_embedding_with_cache(self, image_path):
        cached_embedding = self.embedding_cache.get(image_path)
        if cached_embedding is not None:
            return cached_embedding
        embedding = get_face_embedding(self.face_model, image_path)
        self.embedding_cache.put(image_path, embedding)
        return embedding

    def get_landmarks(self, image_path):
        """
        使用 MTCNN 检测图像的人脸关键点。
        返回一个字典，包含：left_eye, right_eye, nose, mouth_left, mouth_right
        """
        cached = self.landmark_cache.get(image_path)
        if cached is not None:
            return cached

        img = cv2.imread(image_path)
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.face_detector.detect_faces(img_rgb)
        if detections and len(detections) > 0:
            landmarks = detections[0].get("keypoints", None)
            self.landmark_cache.put(image_path, landmarks)
            return landmarks
        return None

    def normalize_landmarks(self, lm):
        """
        对传入的关键点进行归一化:
          - 计算所有关键点的中心值。
          - 用关键点的最小外接矩形对角线长度作为尺度 factor。
          - 返回归一化后的关键点坐标字典。
        """

        points = []
        for key in self.keys:
            if key in lm:
                points.append(np.array(lm[key]))
        if not points:
            return None

        points = np.stack(points)
        center = np.mean(points, axis=0)
        # 计算所有点到中心的最大距离或使用外接矩形的对角线长度
        min_xy = np.min(points, axis=0)
        max_xy = np.max(points, axis=0)
        scale = np.linalg.norm(max_xy - min_xy)
        if scale == 0:
            scale = 1.0
        norm_lm = {}
        for key in self.keys:
            if key in lm:
                pt = np.array(lm[key])
                norm_lm[key] = (pt - center) / scale
        return norm_lm

    def compute_normalized_landmark_difference(self, lm1, lm2):
        """
        计算归一化后的关键点平均欧式差异：
          - 对 lm1 和 lm2 分别归一化
          - 计算对应关键点间的欧式距离
          - 返回所有关键点距离的平均值
        """
        norm_lm1 = self.normalize_landmarks(lm1)
        norm_lm2 = self.normalize_landmarks(lm2)
        if norm_lm1 is None or norm_lm2 is None:
            return None

        keys = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        distances = []
        for key in keys:
            if key in norm_lm1 and key in norm_lm2:
                pt1 = np.array(norm_lm1[key])
                pt2 = np.array(norm_lm2[key])
                dist = np.linalg.norm(pt1 - pt2)
                distances.append(dist)
        if distances:
            return np.mean(distances)
        return 0

    def evaluate(self, data):
        """
        计算改进的 FaceDivNormalized 指标：
          - sim1 = cosine_similarity(id图, imagewithid图)
          - sim2 = cosine_similarity(id图, imagewithoutid图)
          - 定义相似性因子 similarity_factor = sim1 / (sim2 + ε)
          - 提取并归一化 id 图和 imagewithid 图的人脸关键点，计算 normalized_landmark_diff
          - 使用高斯调控函数对 normalized_landmark_diff 进行调控：
                penalty = exp(-((normalized_landmark_diff - T)**2)/(2 * sigma**2))
          - 最终： facediv = similarity_factor * penalty

        注意：归一化后的 landmark_diff 是无量纲的，能有效反映相对于人脸大小的形变。
        """
        # import pdb;pdb.set_trace()
        data_list = data.datas
        category = data.tags['category']
        epsilon = 1e-6  # 防止除零
        facediv_scores = []

        if category == "imageid":
            for item in tqdm(data_list):
                try:
                    id_path = item['id']
                    imagewithid_path = item["imagewithid"]
                    imagewithoutid_path = item["imagewithoutid"]

                    # 计算人脸嵌入
                    id_embedding = self.get_embedding_with_cache(id_path)
                    embedding_withid = get_face_embedding(self.face_model, imagewithid_path)
                    embedding_withoutid = get_face_embedding(self.face_model, imagewithoutid_path)

                    if id_embedding is None or embedding_withid is None or embedding_withoutid is None:
                        continue

                    sim1 = cosine_similarity(id_embedding, embedding_withid)
                    sim2 = cosine_similarity(id_embedding, embedding_withoutid)

                    similarity_factor = sim1 / (sim2 + epsilon)

                    # 计算归一化后的关键点差异
                    lm_id = self.get_landmarks(id_path)
                    lm_withid = self.get_landmarks(imagewithid_path)
                    if lm_id is None or lm_withid is None:
                        continue

                    normalized_landmark_diff = self.compute_normalized_landmark_difference(lm_id, lm_withid)
                    if normalized_landmark_diff is None:
                        continue

                    # 使用高斯调控函数，目标 T 和 sigma 均基于归一化后数值
                    # 当 normalized_landmark_diff 与理想期望 self.T 的差异较大时，无论这个差距是比 self.T 大还是比 self.T 小，公式中的平方项 ((normalized_landmark_diff - self.T)**2) 值都会很大，从而导致指数函数 exp(-(...)) 的指数部分变得很小。也就是说，整个 penalty 值会非常接近 0，而不是接近 1。
                    penalty = exp(- ((normalized_landmark_diff - self.T) ** 2) / (2 * self.sigma ** 2))

                    facediv_score = similarity_factor * penalty
                    facediv_scores.append(facediv_score)
                except Exception as e:
                    logger.error(f"处理项 {item} 出错：{e}")
                    continue

        if len(facediv_scores) == 0:
            return None
        overall_facediv = np.mean(facediv_scores)
        return overall_facediv


@METRICS.register_module()
class FaceDiv2(FaceDiv1):
    def __init__(self, type=""):
        super().__init__()
        # 设定 normalized_landmark_diff 的基准值（通常计算值约为 0.1）
        self.baseline = 0.1
        self.name = "facediv2"
        # λ 权重，用来调节 normalized_landmark_diff 对指标的影响; 你可以根据实际情况调整此值
        self.lambda_weight = 1.0

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags.get('category', None)
        epsilon = 1e-6  # 防止除零
        scores = []

        if category == "imageid":
            for item in tqdm(data_list):
                try:
                    id_path = item['id']
                    imagewithid_path = item["imagewithid"]
                    imagewithoutid_path = item["imagewithoutid"]

                    # 计算人脸嵌入
                    id_embedding = self.get_embedding_with_cache(id_path)
                    embedding_withid = get_face_embedding(self.face_model, imagewithid_path)
                    embedding_withoutid = get_face_embedding(self.face_model, imagewithoutid_path)

                    if id_embedding is None or embedding_withid is None or embedding_withoutid is None:
                        continue

                    sim1 = cosine_similarity(id_embedding, embedding_withid)
                    sim2 = cosine_similarity(id_embedding, embedding_withoutid)
                    similarity_factor = sim1 / (sim2 + epsilon)

                    # 计算归一化后的关键点差异（衡量可编辑性）
                    lm_id = self.get_landmarks(id_path)
                    lm_withid = self.get_landmarks(imagewithid_path)
                    if lm_id is None or lm_withid is None:
                        continue

                    normalized_landmark_diff = self.compute_normalized_landmark_difference(lm_id, lm_withid)
                    if normalized_landmark_diff is None:
                        continue

                    # 指标设计：
                    # facediv = similarity_factor * (1 + λ * (normalized_landmark_diff / baseline))
                    # 当 normalized_landmark_diff 增大时（编辑性更高），整体指标值也会随之升高，
                    # 同时保持 similarity_factor 的主体地位，以保证身份相似度的重要性。
                    facediv = similarity_factor * (
                            1.0 + self.lambda_weight * (normalized_landmark_diff / self.baseline))
                    scores.append(facediv)
                except Exception as e:
                    logger.error(f"Error processing item {item}: {e}")
                    continue

        if not scores:
            return None
        overall_score = np.mean(scores)
        return overall_score
