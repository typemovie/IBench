import cv2
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from mtcnn import MTCNN
# from .posediv import MTCNNDet

from utils.log import logger
from utils.builder import METRICS
from utils.utils import LRUCache


@METRICS.register_module()
class LandmarkDiff(object):
    def __init__(self, type=""):
        self.name = "landmarkdiff"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化 MTCNN 检测器，用于提取人脸关键点。
        # 注意：不要使用 stages="face_detection_only" ，否则不会提取关键点。
        # self.face_detector = MTCNNDet().mtcnn_model
        self.face_detector = MTCNN()
        self.landmark_cache = LRUCache(capacity=20)

        self.keys = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]

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
            return 0

        distances = []
        for key in self.keys:
            if key in norm_lm1 and key in norm_lm2:
                pt1 = np.array(norm_lm1[key])
                pt2 = np.array(norm_lm2[key])
                dist = np.linalg.norm(pt1 - pt2)
                distances.append(dist)
        if distances:
            return np.mean(distances)
        return 0

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags.get('category', None)

        landmarkdiff_scores_list = []
        if category == "imageid":
            for data_item in tqdm(data_list):
                try:
                    id_path = data_item['id']
                    imagewithid_path = data_item["imagewithid"]

                    lm_id = self.get_landmarks(id_path)
                    lm_withid = self.get_landmarks(imagewithid_path)
                    if lm_id is None or lm_withid is None:
                        continue

                    normalized_landmark_diff = self.compute_normalized_landmark_difference(lm_id, lm_withid)
                    landmarkdiff_scores_list.append(normalized_landmark_diff)
                except Exception as e:
                    logger.error(f"处理项 {data_item} 出错：{e}")
                    continue

        cur_avg = np.mean(landmarkdiff_scores_list) if landmarkdiff_scores_list else 0
        return cur_avg
