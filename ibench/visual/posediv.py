import os
import cv2
import uuid
import numpy as np
from PIL import Image
from math import cos, sin
from pathlib import Path
from datetime import datetime
from utils.builder import VISUAL
from utils.compat import config

SAVE_RESULTS = config.save_results


@VISUAL.register_module()
class VisualPoseDiv(object):
    def __init__(self, tdx=None, tdy=None, size=100, type=''):
        self.tdx = tdx
        self.tdy = tdy
        self.size = size

    def visual(self, img, yaw, pitch, roll, file):
        if isinstance(img, Image.Image):
            img = np.asarray(img)[:, :, ::-1]
            # img = cv2.resize(img, (224, 224))[:, :, ::-1]
            img = img.astype(np.uint8).copy()
        # elif isinstance(img, str):
        #     img = cv2.imread(img)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     img = cv2.resize(img, (224, 224))

        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if self.tdx != None and self.tdy != None:
            tdx = self.tdx
            tdy = self.tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = self.size * (cos(yaw) * cos(roll)) + tdx
        y1 = self.size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = self.size * (-cos(yaw) * sin(roll)) + tdx
        y2 = self.size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = self.size * (sin(yaw)) + tdx
        y3 = self.size * (-cos(yaw) * sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

        save_path = os.path.join(SAVE_RESULTS, f'poseidv/{file}')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_img = os.path.join(save_path, f'{timestamp}_{str(uuid.uuid4())}.jpg')
        cv2.imwrite(save_img, img)
