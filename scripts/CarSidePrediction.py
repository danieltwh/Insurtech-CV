import torch
import numpy as np

from typing import List, Tuple
from typing_extensions import Literal
from .typedefs import DetectionPrediction, Model, Image, npArray

class YoloPrediction(DetectionPrediction):
    """ A Class representing a Yolov5 prediction

    Attributes
    ----------
    boundingBox: BoundingBox
        BoundingBox
    confidence: float
        Prediction confidence, ranges from 0. to 1.
    name: str
        Prediction name
    """
    pass

class CarSidePrediction(YoloPrediction):
    """ A Class representing a car side prediction

    Attributes
    ----------
    boundingBox: BoundingBox
        BoundingBox
    confidence: float
        Prediction confidence, ranges from 0. to 1.
    name: Literal["Front", "Back", "Side"]
        Prediction name
    """
    name: Literal["Front", "Back", "Side"]

class YoloModel(Model[CarSidePrediction]):
    def __init__(self, path: str = "./best.pt", **kwargs) -> None:
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path = path, force_reload=True)
        except Exception as e: 
            if ("force_reload=True" in str(e)) :
                raise ValueError("Unable to load model. Received path is: {}".format(path))
            raise e
        
        self.model.conf = kwargs["conf"] if "conf" in kwargs else 0.1
        self.model.iou = kwargs["iou"]  if "iou" in kwargs else 0.45


    def predict_batch(self, imgs:List[Image]) -> Tuple[npArray[Image], npArray[Image], List[CarSidePrediction]]:
        results = self.model(imgs)
        
        original_imgs : npArray[Image]  = np.copy(results.imgs)
        processed_imgs: npArray[Image]  = results.render()
        coords: List[CarSidePrediction] = results.pandas().xyxy[0].to_json(orient="records") # Contains coordinates of the detection
        return original_imgs, processed_imgs, coords

    def predict_single(self, img:Image) -> Tuple[Image, Image, List[CarSidePrediction]]:
        original_imgs, processed_imgs, coords = self.predict_batch([img])
        return original_imgs[0], processed_imgs[0], coords

if __name__ == "__main__":
    # Loading image for demonstration
    from PIL import Image
    img = np.asarray(Image.open('1.jpg'))
    
    # The (only) two lines needed
    model = YoloModel("./best.pt")
    original, processed, coords = model.predict_single(img)

    # Printing coords to show correctness
    print(coords)

    # Saving images to show correctness
    original = Image.fromarray(original)
    processed = Image.fromarray(processed)

    original.save("yolo_original.jpg")
    processed.save("yolo_processed.jpg")
