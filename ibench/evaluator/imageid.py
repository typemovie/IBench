import os
import json
import time
from pathlib import Path
from datetime import datetime
from ibench.instances.data import DataInstance
from utils.builder import build_metrics, build_visual
from utils.utils import write_results
from utils.log import logger
from ibench.metrics import *
from ibench.visual import *
from utils.compat import config

ENABLE_TIMING_STATS = config.enable_timing_stats
SAVE_RESULTS = config.save_results

timing_stats = {
    "model_load": 0.0,
    "fid": 0.0,
    "aesthetic": 0.0,
    "imaging_quality": 0.0,
    "facesim": 0.0,
    "clip-i": 0.0,
    "clip-t": 0.0
}


class ImageidEvaluator(object):

    def __init__(self, config, data_file):
        self.config = config

        # data -----------------------------------------------
        logger.info(f"Loading {data_file}!!!")
        self.data = json.load(open(data_file, "r", encoding="utf-8"))
        self.instances = DataInstance(self.data)

        # model ------------------------------------------------
        model_load_start = time.time() if ENABLE_TIMING_STATS else None

        self.evaluator = config.evaluator
        self.fid = self.evaluator.metrics.get("fid", None)
        self.aesthetic = self.evaluator.metrics.get("aesthetic", None)
        self.imaging_quality = self.evaluator.metrics.get("imaging_quality", None)
        self.facesim = self.evaluator.metrics.get("facesim", None)
        self.antifacesim = self.evaluator.metrics.get("antifacesim", None)
        self.clipi = self.evaluator.metrics.get("clipi", None)
        self.clipt = self.evaluator.metrics.get("clipt", None)
        self.dino = self.evaluator.metrics.get("dino", None)
        self.dreamsim = self.evaluator.metrics.get("dreamsim", None)
        self.fgis = self.evaluator.metrics.get("fgis", None)
        self.posediv = self.evaluator.metrics.get("posediv", None)
        self.exprdiv = self.evaluator.metrics.get("exprdiv", None)
        self.gpt = self.evaluator.metrics.get("gpt", None)
        self.landmarkdiff = self.evaluator.metrics.get("landmarkdiff", None)
        self.facediv1 = self.evaluator.metrics.get("facediv1", None)
        self.facediv2 = self.evaluator.metrics.get("facediv2", None)

        if self.fid:
            self.FIDScore = build_metrics(self.fid, **self.fid)
        if self.aesthetic:
            self.AestheticScore = build_metrics(self.aesthetic, **self.aesthetic)
        if self.imaging_quality:
            self.ImagingQuality = build_metrics(self.imaging_quality, **self.imaging_quality)
        if self.facesim:
            self.FaceSim = build_metrics(self.facesim, **self.facesim)
        if self.antifacesim:
            self.AntiFaceSim = build_metrics(self.antifacesim, **self.antifacesim)
        if self.clipi:
            self.ClipI = build_metrics(self.clipi, **self.clipi)
        if self.clipt:
            self.ClipT = build_metrics(self.clipt, **self.clipt)
        if self.dino:
            self.Dino = build_metrics(self.dino, **self.dino)
        if self.dreamsim:
            self.DreamSim = build_metrics(self.dreamsim, **self.dreamsim)
        if self.fgis:
            self.FGIS = build_metrics(self.fgis, **self.fgis)
        if self.posediv:
            self.Posediv = build_metrics(self.posediv, **self.posediv)
        if self.exprdiv:
            self.Exprdiv = build_metrics(self.exprdiv, **self.exprdiv)
        if self.landmarkdiff:
            self.LandmarkDiff = build_metrics(self.landmarkdiff, **self.landmarkdiff)
        if self.facediv1:
            self.FaceDiv1 = build_metrics(self.facediv1, **self.facediv1)
        if self.facediv2:
            self.FaceDiv2 = build_metrics(self.facediv2, **self.facediv2)
        if self.gpt:
            self.GPT = build_metrics(self.gpt, **self.gpt)

        if ENABLE_TIMING_STATS:
            timing_stats['model_load'] = time.time() - model_load_start

        # visual ----------------------------------------------------------------------------------
        self.visual = self.evaluator.visual
        self.visualposediv = self.visual.get("visualposediv", None)

        if self.visualposediv:
            self.visualposediv = build_visual(self.visualposediv, **self.visualposediv)

        # save --------------------------------------------------------
        Path(SAVE_RESULTS).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.save_results_config = os.path.join(SAVE_RESULTS, f"{timestamp}_config.py")
        write_results(self.save_results_config, self.config.text)
        write_results(self.save_results_config, data_file)

    def evaluate(self):
        # import pdb;pdb.set_trace()
        # 1.T2i----------------------------------------------------------------------------------
        if self.fid:
            result = self.FIDScore.evaluate(self.instances)
            write_results(self.save_results_config, f"fid: {result}")
            logger.info(f"===================== fid: {result}")

        if self.aesthetic:
            result = self.AestheticScore.evaluate(self.instances)
            write_results(self.save_results_config, f"aesthetic: {result}")
            logger.info(f"===================== aesthetic: {result}")

        if self.imaging_quality:
            result = self.ImagingQuality.evaluate(self.instances)
            write_results(self.save_results_config, f"imaging_quality: {result}")
            logger.info(f"===================== imaging_quality: {result}")

        # 2.Consistency
        if self.facesim:
            result = self.FaceSim.evaluate(self.instances)
            write_results(self.save_results_config, f"facesim: {result}")
            logger.info(f"===================== facesim: {result}")

        if self.antifacesim:
            result = self.AntiFaceSim.evaluate(self.instances)
            write_results(self.save_results_config, f"antifacesim: {result}")
            logger.info(f"===================== antifacesim: {result}")

        if self.clipi:
            result = self.ClipI.evaluate(self.instances)
            write_results(self.save_results_config, f"clipi: {result}")
            logger.info(f"===================== clipi: {result}")

        if self.clipt:
            result = self.ClipT.evaluate(self.instances)
            write_results(self.save_results_config, f"clipt:{result}")
            logger.info(f"===================== clipt:{result}")

        if self.dino:
            result = self.Dino.evaluate(self.instances)
            write_results(self.save_results_config, f"dino:{result}")
            logger.info(f"===================== dino:{result}")

        if self.dreamsim:
            result = self.DreamSim.evaluate(self.instances)
            write_results(self.save_results_config, f"dreamsim:{result}")
            logger.info(f"===================== dreamsim:{result}")

        if self.fgis:
            result = self.FGIS.evaluate(self.instances)
            write_results(self.save_results_config, f"fgis:{result}")
            logger.info(f"===================== fgis:{result}")

        if self.posediv:
            result = self.Posediv.evaluate(self.instances, self.visualposediv)
            write_results(self.save_results_config, f"posediv:{result}")
            logger.info(f"===================== posediv:{result}")

        if self.exprdiv:
            result = self.Exprdiv.evaluate(self.instances)
            write_results(self.save_results_config, f"exprdiv:{result}")
            logger.info(f"===================== exprdiv:{result}")

        if self.landmarkdiff:
            result = self.LandmarkDiff.evaluate(self.instances)
            write_results(self.save_results_config, f"landmarkdiff:{result}")
            logger.info(f"===================== landmarkdiff:{result}")

        if self.facediv1:
            result = self.FaceDiv1.evaluate(self.instances)
            write_results(self.save_results_config, f"facediv:{result}")
            logger.info(f"===================== facediv:{result}")

        if self.facediv2:
            result = self.FaceDiv2.evaluate(self.instances)
            write_results(self.save_results_config, f"facediv:{result}")
            logger.info(f"===================== facediv:{result}")

        if self.gpt:
            result = self.GPT.evaluate(self.instances)
            write_results(self.save_results_config, f"gpt:{result}")
            logger.info(f"===================== gpt:{result}")
