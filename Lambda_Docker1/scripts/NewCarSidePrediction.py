from .predictions.YoloPrediction import YoloPrediction
from .models.YoloModel import YoloModel

from typing_extensions import Literal

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

class CarSideModel(YoloModel[CarSidePrediction]):
    def __init__(self, path: str = "./best.pt", **kwargs) -> None:
        super().__init__(path=path, **kwargs)
        self.model.conf = kwargs["conf"] if "conf" in kwargs else 0.1
        self.model.iou = kwargs["iou"]  if "iou" in kwargs else 0.45