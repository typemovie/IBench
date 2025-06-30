import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from math import exp, log

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
    if faces and len(faces) > 0:
        return faces[0].embedding
    return None

def cosine_similarity(a, b):
    """
    计算余弦相似度
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0
    a_normalized = a / a_norm
    b_normalized = b / b_norm
    return np.dot(a_normalized, b_normalized)

@METRICS.register_module()
class FaceDivLoss(object):
    """
    FaceDivLoss 损失函数旨在同时衡量生成图像与身份图（id）的相似度及
    可编辑性。假设我们的生成模型接收 id 图 + text 作为输入，输出 imagewithid，
    希望输出图在尽量保留 id 特征的同时又具备较高的可编辑性。
    
    衡量准则：
      1. 身份保真度：直接用 id 图与生成图之间的余弦相似度 sim
         为了让相似度越高，损失越小，使用 -log(similarity) 作为惩罚项。
      2. 可编辑性：通过提取人脸关键点计算归一化后的平均欧氏距离 diff，
         当 diff 接近预定目标 T 时，说明生成图具有理想的编辑幅度。
         我们通过固定高斯函数奖励，该奖励值为：
             R = exp( - ((diff - T)**2) / (2 * sigma**2) )
         为使奖励值越高损失越小，我们在损失函数中采用：
             -log( R ) = ((diff - T)**2) / (2 * sigma**2)
         
    综上，FaceDivLoss 定义为：
         Loss = - log( cosine_similarity(embedding_id, embedding_generated) + eps )
                + ((normalized_landmark_diff - T)**2) / (2 * sigma**2)
                
    其中 eps 用于防止取对数时出现数值问题。该设计使得：
      - 当生成图与 id 图在嵌入空间内越相似时，第一项损失越小；
      - 当生成图在人脸关键点的编辑变化差异越接近理想目标 T 时，
        第二项损失越小，从而同时鼓励保真与编辑性。
    """
    def __init__(self):
        self.name = "facediv_loss"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载用于计算嵌入的人脸模型
        self.face_model = FaceModelLoader().get_model()
        self.embedding_cache = LRUCache(capacity=20)
        
        # 初始化 MTCNN 检测器用于提取人脸关键点
        self.face_detector = MTCNN()
        self.landmark_cache = LRUCache(capacity=20)
        self.keys = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]

        # 固定高斯函数参数: 理想的关键点归一化差异及宽度
        self.T = 0.2
        self.sigma = 0.1
        self.eps = 1e-6  # 防止取对数时数值为零

    def get_embedding_with_cache(self, image_path):
        cached = self.embedding_cache.get(image_path)
        if cached is not None:
            return cached
        embedding = get_face_embedding(self.face_model, image_path)
        self.embedding_cache.put(image_path, embedding)
        return embedding

    def get_landmarks(self, image_path):
        """
        使用 MTCNN 检测图像的人脸关键点，
        返回一个包含 left_eye, right_eye, nose, mouth_left, mouth_right 的字典。
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
        将关键点归一化：
         - 计算关键点中心
         - 使用关键点最小外接矩形对角线长度作为尺度
         - 返回归一化后的关键点字典
        """
        points = []
        for key in self.keys:
            if key in lm:
                points.append(np.array(lm[key]))
        if not points:
            return None
        points = np.stack(points)
        center = np.mean(points, axis=0)
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
        计算归一化后的平均欧式距离，即关键点差异
        """
        norm_lm1 = self.normalize_landmarks(lm1)
        norm_lm2 = self.normalize_landmarks(lm2)
        if norm_lm1 is None or norm_lm2 is None:
            return None
        
        distances = []
        for key in self.keys:
            if key in norm_lm1 and key in norm_lm2:
                pt1 = np.array(norm_lm1[key])
                pt2 = np.array(norm_lm2[key])
                distances.append(np.linalg.norm(pt1 - pt2))
        if distances:
            return np.mean(distances)
        return 0

    def compute_loss(self, id_image_path, generated_image_path):
        """
        计算 FaceDivLoss 损失。
        参数:
          id_image_path: 原始身份图
          generated_image_path: 生成的带编辑效果的图（imagewithid）
        返回:
          loss 值（越小越好）
        """
        # 获取人脸嵌入
        id_embedding = self.get_embedding_with_cache(id_image_path)
        gen_embedding = get_face_embedding(self.face_model, generated_image_path)
        if id_embedding is None or gen_embedding is None:
            logger.error("无法获得图像嵌入")
            return None
        
        # 计算身份相似度
        sim = cosine_similarity(id_embedding, gen_embedding)
        sim = max(sim, self.eps)  # 避免小于 eps
        
        # 使用 -log(similarity) 作为身份损失
        identity_loss = -log(sim)
        
        # 获得并归一化关键点
        lm_id = self.get_landmarks(id_image_path)
        lm_gen = self.get_landmarks(generated_image_path)
        if lm_id is None or lm_gen is None:
            logger.error("无法获得图像关键点")
            return None
        
        normalized_diff = self.compute_normalized_landmark_difference(lm_id, lm_gen)
        if normalized_diff is None:
            logger.error("无法计算关键点差异")
            return None
        
        # 可编辑性损失，期望归一化关键点差异接近 T
        editability_loss = ((normalized_diff - self.T) ** 2) / (2 * self.sigma ** 2)
        
        # 总损失：身份损失与编辑性损失之和
        total_loss = identity_loss + editability_loss
        return total_loss

# 如果需要，可以增加一个简单调用示例
if __name__ == "__main__":
    # 示例路径，请根据实际情况修改
    id_path = "path/to/id_image.jpg"
    gen_path = "path/to/generated_image.jpg"
    
    loss_fn = FaceDivLoss()
    loss_value = loss_fn.compute_loss(id_path, gen_path)
    if loss_value is not None:
        print(f"FaceDivLoss: {loss_value}")
    else:
        print("无法计算 FaceDivLoss")