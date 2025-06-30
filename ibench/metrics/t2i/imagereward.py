import torch
import ImageReward as RM
from utils.log import logger
from utils.compat import config
from utils.builder import METRICS

@METRICS.register_module()
class ImageReward(object):
    def __init__(self,type=""):
        self.name = "imagereward"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ----------------------------------------------------------------------------------------------------------------
        logger.info(f"Loading {self.name} model!!!!")
        # self.model =

    def load_model(self):
        model = RM.load()


