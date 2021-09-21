from typing import List, Literal, Tuple, TypedDict, Union
import torch
import numpy as np

from PredictionType import DetectionPrediction

class YoloPrediction(DetectionPrediction):
    pass

class CarSidePrediction(YoloPrediction):
    name: Literal["Front", "Back", "Side"]

class YoloModel:
    def __init__(self, path: str = "./best.pt", **kwargs) -> None:
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path = path, force_reload=True)
        except Exception as e: 
            if ("force_reload=True" in str(e)) :
                raise ValueError("Unable to load model. Received path is: {}".format(path))
            raise e
        
        self.model.conf = kwargs["conf"] if "conf" in kwargs else 0.25  # confidence threshold [0., 1.]
        self.model.iou = kwargs["iou"]  if "iou" in kwargs else 0.45 # NMS IoU threshold [0., 1.]


    def predict_batch(self, imgs:List[np.ndarray]) -> Tuple[np.ndarray, List[CarSidePrediction]]:
        results = self.model(imgs)
        
        original_imgs = results.imgs
        coords = results.pandas().xyxy[0].to_json(orient="records") # Contains coordinates of the detection
        return original_imgs, coords

    def predict_single(self, img:np.ndarray) -> Tuple[np.ndarray, List[CarSidePrediction]]:
        original_imgs, coords = self.predict_batch([img])
        return original_imgs[0], coords

if __name__ == "__main__":
    model = YoloModel()

    import cv2
    img = cv2.imread("1.jpg")
    
    original, coords = model.predict_single(img)
    print(coords)
    cv2.imwrite("original.jpg", original)
