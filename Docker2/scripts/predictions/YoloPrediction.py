from .Prediction import DetectionPrediction
from ..typedefs import BoundingBox

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
    def __init__(self, bbox: BoundingBox, conf: float, name: str) -> None:
        super().__init__(bbox, conf, name)