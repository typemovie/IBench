import sys

sys.path.append("/home/gdli7/IBench/")

import os
import json
import cv2
from insightface.app import FaceAnalysis
from utils.compat import config
from tqdm import tqdm

FaceSim_MODEL_PATH = config.metrics.imageid.facesim.face_detection_model_path

# 初始化FaceAnalysis
app = FaceAnalysis(name="antelopev2", root=FaceSim_MODEL_PATH,
                   providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=1, det_size=(640, 640))


def predict_gender(image_path):
    # 读取图片
    image = cv2.imread(image_path)
    faces = app.get(image)

    if len(faces) > 0:
        # 只处理第一个检测到的人脸
        face = faces[0]
        gender = 'female' if face.gender == 0 else 'male'
        return gender
    else:
        return 'unknown'


def generate_image_data(image_folder):
    data = {"datas": [], "tags": {"category": "imageid"}}
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # import pdb;
    # pdb.set_trace()
    for num, image_file in tqdm(enumerate(image_files)):
        image_path = os.path.join(image_folder, image_file)
        gender = predict_gender(image_path)
        data["datas"].append({
            "num": num,
            "id": image_path,
            "gender": gender
        })

    return data


if __name__ == "__main__":
    config1 = {
        "image_folder_path": "/home/gdli7/IBench/data/images/unsplash_50/",
        "output_file": "/home/gdli7/IBench/data/images/unsplash50_images.json"
    }
    config2 = {
        "image_folder_path": "/home/gdli7/IBench/data/images/chineseid/",
        "output_file": "/home/gdli7/IBench/data/images/chineseid_images.json"
    }

    config3 = {
        "image_folder_path": "/home/gdli7/IBench/data/images/generateid/",
        "output_file": "/home/gdli7/IBench/data/images/generateid_images.json"
    }

    config4 = {
        "image_folder_path": "/home/gdli7/IBench/data/images/mystyleid/",
        "output_file": "/home/gdli7/IBench/data/images/mystyleid_images.json"
    }

    config = config4
    # 示例使用
    image_folder_path = config["image_folder_path"]
    image_data = generate_image_data(image_folder_path)

    # 将数据保存为JSON文件
    output_file = config["output_file"]
    with open(output_file, "w") as f:
        json.dump(image_data, f, indent=2)

    print(f"JSON数据已保存到 {output_file}")
