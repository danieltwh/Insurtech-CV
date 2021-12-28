from .Prediction import DetectionPrediction
from ..typedefs import BoundingBox, Mask

class MrcnnPrediction(DetectionPrediction):
    """ A Class representing an Mrcnn prediction

    Attributes
    ----------
    mask: Mask
        Mask of detected object
    bbox: BoundingBox
        Bounding box of detected object
    confidence: float
        Prediction confidence, ranges from 0. to 1.
    name: str
        Prediction name
    """
    mask: Mask
    def __init__(self, mask: Mask, bbox: BoundingBox, conf: float, name: str) -> None:
        super().__init__(bbox, conf, name)
        self.mask = mask
    
    def getMask(self) -> Mask: 
        return self.mask