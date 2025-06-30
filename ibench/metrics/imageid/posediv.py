import cv2
import numpy as np
import torch
import uuid
import torchvision
from .posediv_hopenet import Hopenet

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms
from mtcnn import MTCNN
from utils.log import logger
from datetime import datetime

from utils.builder import METRICS
from utils.compat import config
from utils.utils import LRUCache

HOPENET_MODEL_PATH = config.metrics.imageid.posediv.hopenet_model_path
YAW_THRESHOLD = config.metrics.imageid.posediv.yaw_threshold
PITCH_THRESHOLD = config.metrics.imageid.posediv.pitch_threshold
ROLL_THRESHOLD = config.metrics.imageid.posediv.roll_threshold


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


class MTCNNDet:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'mtcnn_model'):
            logger.info("Loading mtcnn model!!!")
            self.mtcnn_model = self.load_mtcnn()

    def load_mtcnn(self):
        mtcnn = MTCNN(stages="face_detection_only")
        return mtcnn


class HopeNet:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'device'):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # -------------------------------------------------------------------------------------------------------------------
            logger.info("Loading hopenet-posediv model!!!!")
            self.model = self.load_model()

            self.transformations = transforms.Compose([transforms.Resize(224),
                                                       transforms.CenterCrop(
                                                           224), transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])

    def load_model(self):
        model = Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        state_dict = torch.load(HOPENET_MODEL_PATH, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(self.device)
        return model


@METRICS.register_module()
class PoseDiv(object):
    def __init__(self, type=""):
        self.name = "posediv"
        # self.uuid = {str(uuid.uuid4())}
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # import pdb;pdb.set_trace()
        Hopenet = HopeNet()
        self.model = Hopenet.model
        self.transformations = Hopenet.transformations
        mtcnn_model = MTCNNDet()
        self.face_detector = mtcnn_model.mtcnn_model
        self.device = Hopenet.device
        self.embedding_cache = LRUCache(capacity=20)

        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).to(self.device)

    def predict(self, image):
        image = self.transformations(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        yaw, pitch, roll = self.model(image)

        # 特征归一化
        yaw_predicted = softmax_temperature(yaw.data, 1)
        pitch_predicted = softmax_temperature(pitch.data, 1)
        roll_predicted = softmax_temperature(roll.data, 1)

        # 将期望变换到连续值的读书
        yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * self.idx_tensor, 1).cpu() * 3 - 99

        yaw_predicted = yaw_predicted.item()
        pitch_predicted = pitch_predicted.item()
        roll_predicted = roll_predicted.item()
        return yaw_predicted, pitch_predicted, roll_predicted

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

    def get_embedding_with_cache(self, image_path, visualposediv):
        cached_embedding = self.embedding_cache.get(image_path)
        if cached_embedding is not None:
            return cached_embedding

        region = self.face_detect(image_path)
        image = Image.open(image_path).convert('RGB')
        if region is not None:
            x, y, w, h = region
            image_face = image.crop((int(x - 30), int(y - 30), int(x + w + 30), int(y + h + 30)))
        else:
            image_face = image
        image_yaw, image_pitch, image_roll = self.predict(image_face)
        visualposediv.visual(image_face, image_yaw, image_pitch, image_roll, self.timestamp)

        self.embedding_cache.put(image_path, (image_yaw, image_pitch, image_roll))
        return image_yaw, image_pitch, image_roll

    def evaluate(self, data, visualposediv):
        data_list = data.datas
        category = data.tags['category']

        yaw_scores_list = []
        pitch_scores_list = []
        roll_scores_list = []
        # import pdb;
        # pdb.set_trace()
        if category == "imageid":
            for data in tqdm(data_list):
                try:
                    id = data['id']
                    id_yaw, id_pitch, id_roll = self.get_embedding_with_cache(id, visualposediv)

                    imagewithid = data['imagewithid']
                    region = self.face_detect(imagewithid)
                    imagewithid = Image.open(imagewithid).convert('RGB')
                    if region is not None:
                        x, y, w, h = region
                        imagewithid_face = imagewithid.crop(
                            (int(x - 30), int(y - 30), int(x + w + 30), int(y + h + 30)))
                    else:
                        imagewithid_face = imagewithid
                    imagewithid_yaw, imagewithid_pitch, imagewithid_roll = self.predict(imagewithid_face)
                    visualposediv.visual(imagewithid_face, imagewithid_yaw, imagewithid_pitch, imagewithid_roll,
                                         self.timestamp)

                    yaw_error = abs(id_yaw - imagewithid_yaw)
                    pitch_error = abs(id_pitch - imagewithid_pitch)
                    roll_error = abs(id_roll - imagewithid_roll)

                    yaw_scores_list.append(yaw_error)
                    pitch_scores_list.append(pitch_error)
                    roll_scores_list.append(roll_error)
                except Exception as e:
                    logger.error(f"处理项 {data} 出错：{e}")
                    continue

        yaw_avg = np.mean(yaw_scores_list)
        pitch_avg = np.mean(pitch_scores_list)
        roll_avg = np.mean(roll_scores_list)
        return yaw_avg, pitch_avg, roll_avg
