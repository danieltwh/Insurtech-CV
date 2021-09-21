import os 
import sys
from typing import List, Literal, Tuple, TypedDict

import numpy as np

from PredictionType import DetectionPrediction

sys.path.append(os.path.abspath('../Mask_RCNN'))
from mrcnn.config import Config
from mrcnn.model import MaskRCNN

'''
TODO: UNTESTED CODE
'''
class MRCNNPrediction(DetectionPrediction):
    mask: np.ndarray

class DamagePrediction(MRCNNPrediction):
    name: Literal["BG", "scratches", "dents"]

class MRCNNModel:
    class PredictionConfig(Config):
        NAME = "damage_cfg"
        NUM_CLASSES = 1 + 2 # Background + Everything else
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Non-maximum suppression threshold for detection
        DETECTION_MIN_CONFIDENCE: float    
        DETECTION_NMS_THRESHOLD: float

        def __init__(self, conf: float, iou: float) -> None:
            super().__init__()
            self.DETECTION_MIN_CONFIDENCE = conf
            self.DETECTION_NMS_THRESHOLD = iou

    def __init__(self, path: str = "best.h5", **kwargs) -> None:
        conf : float = kwargs["conf"] if "conf" in kwargs else 0.7  # confidence threshold [0., 1.]
        iou : float = kwargs["iou"]  if "iou" in kwargs else 0.3    # NMS IoU threshold [0., 1.]

        self.cfg = self.PredictionConfig(conf, iou)
        self.model = MaskRCNN(mode="inference", model_dir="./", config=self.cfg)
        self.model.load_weights(path, by_name=True)

    def predict_batch(self, imgs:List[np.ndarray]) ->  Tuple[np.ndarray, List[DamagePrediction]]:
        # TODO: change imgs from List[np.ndarray] to standard np.ndarray
        scaled_imgs = mold_image(imgs, self.cfg)
        results = self.model.detect(scaled_imgs, 0)

        original_imgs = np.stack(imgs, axis=0 )

        return original_imgs, results

    def predict_single(self, img:np.ndarray) -> Tuple[np.ndarray, List[DamagePrediction]]:
        original_imgs, coords = self.predict_batch([img])
        return original_imgs[0], coords

if __name__ == "__main__":
    model = MRCNNModel()

    import cv2
    img = cv2.imread("1.jpg")
    
    original, coords = model.predict_single(img)
    print(coords)
    cv2.imwrite("original.jpg", original)