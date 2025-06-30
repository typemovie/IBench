import os
import torch
import shutil
import pathlib
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg
from pathlib import Path
from omegaconf import OmegaConf

from .fid_inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as TF

from utils.builder import METRICS
from utils.log import logger
from utils.compat import config

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}

# FID_DEVICE = config.metrics.t2i.fid.fid_device
TARGET_WITHID = config.metrics.t2i.fid.target_withid
TARGET_WITHOUTID = config.metrics.t2i.fid.target_withoutid


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
            mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
            sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
                  "fid calculation produces singular product; "
                  "adding %s to diagonal of cov estimates"
              ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_activations(files, model, batch_size=50, dims=2048, device="cpu", num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()
    if batch_size > len(files):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx: start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(
        files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device, num_workers):
    if path.endswith(".npz"):
        with np.load(path) as f:
            m, s = f["mu"][:], f["sigma"][:]
    else:
        path = pathlib.Path(path)
        files = sorted(
            [file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))]
        )
        m, s = calculate_activation_statistics(
            files, model, batch_size, dims, device, num_workers
        )

    return m, s


def download_image(url, save_path):
    # 下载图片并保存到指定路径
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    else:
        print(f"Failed to download image from {url}")


@METRICS.register_module()
class FIDScore(object):
    def __init__(self, dims=2048, batch_size=50, num_workers=1, type=""):
        self.name = "fid"

        # --------------------------------------------
        self.dims = dims
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # -----------------------------------------------
        logger.info(f"Loading {self.name} Inception model!!!")
        self.model = self.load_model()

    def load_model(self):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        model = InceptionV3([block_idx]).to("cuda")
        return model

    def calculate_fid_given_paths(self, image, target):
        m1, s1 = compute_statistics_of_path(image, self.model, self.batch_size, self.dims, self.device,
                                            self.num_workers)
        m2, s2 = compute_statistics_of_path(target, self.model, self.batch_size, self.dims, self.device,
                                            self.num_workers)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value

    def evaluate(self, data):
        # 1.对输入的data进行粗粒
        category = data.tags['category']
        if category == "imageid":
            data_list = data.datas

            Path(TARGET_WITHID).mkdir(parents=True, exist_ok=True)
            Path(TARGET_WITHOUTID).mkdir(parents=True, exist_ok=True)

            for item in data_list:
                # 处理imagewithid图片
                if item['imagewithid'].startswith('http'):
                    # 下载图片
                    save_path_with_id = os.path.join(TARGET_WITHID, os.path.basename(item['imagewithid']))
                    download_image(item['imagewithid'], save_path_with_id)
                else:
                    # 复制图片
                    shutil.copy(item['imagewithid'], TARGET_WITHID)

                # 处理imagewithoutid图片
                if item['imagewithoutid'].startswith('http'):
                    # 下载图片
                    save_path_without_id = os.path.join(TARGET_WITHOUTID, os.path.basename(item['imagewithoutid']))
                    download_image(item['imagewithoutid'], save_path_without_id)
                else:
                    # 复制图片
                    shutil.copy(item['imagewithoutid'], TARGET_WITHOUTID)
        else:
            raise ValueError("just spport imageid and t2i fid evaluate!!")

        # 处理成两个地址来计算，批量计算图片
        fid = self.calculate_fid_given_paths(TARGET_WITHID, TARGET_WITHOUTID)
        return fid
