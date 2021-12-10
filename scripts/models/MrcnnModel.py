from scripts.predictions.MrcnnPrediction import MrcnnPrediction
from scripts.models.Model import Model

from typing import List, Tuple, NoReturn, TypeVar
from scripts.typedefs import Image, BoundingBox, Mask

import numpy as np

import os, sys
sys.path.append(os.path.abspath('./Mask_RCNN'))
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, mold_image
from mrcnn.utils import extract_bboxes

T = TypeVar("T", bound=MrcnnPrediction)
class MrcnnModel(Model[T]):
    class PredictionConfig(Config):
        NAME = "damage_cfg"
        NUM_CLASSES = 1 + 2 # Background + Everything else
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Non-maximum suppression threshold for detection
        DETECTION_MIN_CONFIDENCE: float    
        DETECTION_NMS_THRESHOLD: float

        def __init__(self, conf: float, iou: float) -> NoReturn:
            super().__init__()
            self.DETECTION_MIN_CONFIDENCE = conf
            self.DETECTION_NMS_THRESHOLD = iou

    def __init__(self, path: str = "best.h5", **kwargs) -> NoReturn:
        conf : float = kwargs["conf"] if "conf" in kwargs else 0.7  # confidence threshold [0., 1.]
        iou : float = kwargs["iou"]  if "iou" in kwargs else 0.3    # NMS IoU threshold [0., 1.]

        self.cfg = self.PredictionConfig(conf, iou)
        self.model = MaskRCNN(mode="inference", model_dir="./", config=self.cfg)
        self.model.load_weights(path, by_name=True)

    def _extractBboxes(self, detection: List) -> BoundingBox:
        return BoundingBox(detection[1], detection[0], detection[3], detection[2])

    def predict_batch(self, imgs:List[Image]) ->  Tuple[Image, Image, List[List[T]]]:
        original_imgs = np.stack(imgs, axis=0 )
        scaled_imgs = mold_image(original_imgs, self.cfg)
        detections = self.model.detect(scaled_imgs, 0)

        predictions = []
        for detection in detections:
            rois = detection["rois"]
            class_ids = detection["class_ids"]
            scores = detection["scores"]
            masks: List[Mask] = detection["masks"]
            bboxes = extract_bboxes(masks)

            prediction = []
            for i, class_id in enumerate(class_ids):
                score = scores[i]
                bbox = self._extractBboxes(detection=bboxes[i])
                mask = masks[i]
                prediction.append(MrcnnPrediction(mask, bbox, score, class_id))
            predictions.append(prediction)

        # TODO: update to include processing of images
        return original_imgs, original_imgs, predictions

    def predict_single(self, img:Image) -> Tuple[Image, Image, List[T]]:
        original_imgs, processed_imgs, coords = self.predict_batch([img])
        return original_imgs[0], processed_imgs[0], coords[0]
