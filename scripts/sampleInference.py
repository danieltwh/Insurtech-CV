from typing import List, Literal, Tuple, TypedDict, Union
import torch
import numpy as np

class YoloPrediction(TypedDict):
    # (0, 0) is top-left of the image
    # bottom-right of the image is most positive coordinates
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    confidence: float       # Confidence of prediction
    name: str               # Prediction name

class CarSidePrediction(YoloPrediction):
    name: Literal["Front", "Back", "Side"]

def getModel(path:str = "best.pt", **kwargs):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = path, force_reload=True)
    
    model.conf = kwargs["conf"] if "conf" in kwargs else 0.25  # confidence threshold (0-1)
    model.iou = kwargs["iou"]  if "iou" in kwargs else 0.45 # NMS IoU threshold (0-1)

    return model

def predict_batch(model, imgs:List[np.ndarray]) -> Tuple[np.ndarray, List[CarSidePrediction]]:
    results = model(imgs)
    
    original_imgs = results.imgs
    coords = results.pandas().xyxy[0].to_json(orient="records") # Contains coordinates of the detection
    return original_imgs, coords

def predict_single(model, img:np.ndarray) -> Tuple[np.ndarray, List[CarSidePrediction]]:
    original_imgs, coords = predict_batch(model, [img])
    return original_imgs[0], coords

if __name__ == "__main__":
    model = getModel()

    import cv2

    img = cv2.imread("1.jpg")
    original, coords = predict_single(model, img)
    print(coords)
    cv2.imwrite("original.jpg", original)
