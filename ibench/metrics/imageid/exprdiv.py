import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from .posediv import MTCNNDet
from utils.builder import METRICS
from utils.log import logger
from utils.compat import config
from utils.utils import LRUCache

EXPRESSION_MODEL_PATH = config.metrics.imageid.exprdiv.expression_model_path

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


@METRICS.register_module()
class ExprDiv(object):
    def __init__(self, type=""):
        self.name = "exprdiv"
        mtcnn_model = MTCNNDet()
        self.face_detector = mtcnn_model.mtcnn_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -----------------------------------------------------------------------
        logger.info(f"Loading {self.name} model!!!!")
        self.model = self.load_model()
        self.embedding_cache = LRUCache(capacity=20)

        self.transformations = transforms.Compose([
            transforms.TenCrop(44),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])

    def load_model(self):
        model = VGG("VGG19")
        model.load_state_dict(torch.load(EXPRESSION_MODEL_PATH, weights_only=True)['net'])
        model = model.to(self.device)
        model.eval()
        return model

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

    def predict(self, image):
        image = self.transformations(image)
        ncrops, c, h, w = image.shape
        image = image.view(-1, c, h, w)
        image = image.to(self.device)

        output = self.model(image)
        outputs_avg = output.view(ncrops, -1).mean(0)
        score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)
        return predicted.item()

    def get_embedding_with_cache(self, image_path):
        # 如果缓存中已有该图像路径对应的嵌入，则直接返回
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
        image_expr = self.predict(image_face)

        self.embedding_cache.put(image_path, image_expr)
        return image_expr

    def evaluate(self, data):
        data_list = data.datas
        category = data.tags['category']

        num = 0
        different_expression_count = 0
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
                    # id_expr = self.predict(id_face)
                    id_expr = self.get_embedding_with_cache(id)

                    imagewithid = data['imagewithid']
                    region = self.face_detect(imagewithid)
                    imagewithid = Image.open(imagewithid).convert('RGB')
                    if region is not None:
                        x, y, w, h = region
                        imagewithid_face = imagewithid.crop(
                            (int(x - 30), int(y - 30), int(x + w + 30), int(y + h + 30)))
                    else:
                        imagewithid_face = imagewithid
                    imagewithid_expr = self.predict(imagewithid_face)

                    if id_expr != imagewithid_expr:
                        different_expression_count += 1

                    num += 1
                except Exception as e:
                    logger.error(f"处理项 {data} 出错：{e}")
                    continue

        exprdiv = different_expression_count / num
        return exprdiv
