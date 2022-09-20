from scripts.predictions.YoloPrediction import YoloPrediction
from scripts.models.Model import Model

from typing import Dict, List, Tuple, TypeVar
from scripts.typedefs import BoundingBox, Image, npArray

import json
import numpy as np
import torch
torch.cuda.is_available = lambda : False # Forces to use CPU to prevent segfault

T = TypeVar("T", bound=YoloPrediction)
class YoloModel(Model[T]):
    def __init__(self, path: str = "./best.pt", **kwargs) -> None:
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path = path, force_reload=False)
        except Exception as e: 
            if ("force_reload=True" in str(e)) :
                raise ValueError("Unable to load model. Received path is: {}".format(path))
            raise e
        
        self.model.conf = kwargs["conf"] if "conf" in kwargs else 0.1
        self.model.iou = kwargs["iou"]  if "iou" in kwargs else 0.45


    def _extractBboxes(self, detection: Dict) -> BoundingBox:
            return BoundingBox(detection["xmin"], detection["ymin"], detection["xmax"], detection["ymax"])

    def predict_batch(self, imgs:List[Image]) -> Tuple[npArray[Image], npArray[Image], List[List[T]]]:
        results = self.model(imgs)
        
        # original_imgs : npArray[Image]  = np.copy(results.imgs)
        original_imgs : npArray[Image]  = np.copy(imgs)
        processed_imgs: npArray[Image]  = results.render()
        detections = results.pandas().xyxy # Contains coordinates of the detection

        predictions = []
        for detection in detections:
            detection = json.loads(detection.to_json(orient="records"))
            prediction = []
            for det in detection:
                class_id = det["name"]
                score = det["confidence"]
                bbox = self._extractBboxes(detection=det)
                prediction.append(YoloPrediction(bbox, score, class_id))
            predictions.append(prediction)

        return original_imgs, processed_imgs, predictions

    def predict_single(self, img:Image) -> Tuple[Image, Image, List[T]]:
        original_imgs, processed_imgs, predictions = self.predict_batch([img])
        
        return original_imgs[0], processed_imgs[0], predictions[0]