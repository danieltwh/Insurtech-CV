class BoundingBox:
    """ A Class representing a bounding box

    A BoundingBox must obey the following format: 
    - (0,0) coordinate corresponds to top-left of image
    - (xmax, ymax) coordinate corresponds to bottom-right of image

    Attributes
    ----------
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class DetectionPrediction:
    """ Parent class of all model predictions.

    Any ModelPrediction must inherit from DetectionPrediction. This class defines
    the expected output of a Model. 

    Attributes
    ----------
    boundingBox: BoundingBox
        BoundingBox
    confidence: float
        Prediction confidence, ranges from 0. to 1.
    name: str
        Prediction name
    """
    boundingBox: BoundingBox
    confidence: float   
    name: str

    def __init__(self, bbox: BoundingBox, conf: float, name: str) -> None:
        self.boundingBox = bbox
        self.confidence = conf
        self.name = name
    
    def getBoundingBox(self) -> BoundingBox:
        return self.boundingBox
    
    def getConfidence(self) -> float:
        return max(min(self.confidence, 1.), 0.)
    
    def getName(self) -> str:
        return self.name
