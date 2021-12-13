
from .predictions.YoloPrediction import YoloPrediction
from .models.YoloModel import YoloModel

from .predictions.MrcnnPrediction import MrcnnPrediction
from .models.MrcnnModel import MrcnnModel

from .typedefs import Image, Mask, BoundingBox
from typing import List, Tuple
from typing_extensions import Literal


class DamageSegmentationPrediction(MrcnnPrediction):
    """ A Class representing a damage segmentation prediction

    Attributes
    ----------
    mask: np.ndarray
    boundingBox: BoundingBox
        BoundingBox
    confidence: float
        Prediction confidence, ranges from 0. to 1.
    """
    name: Literal["BG", "scratches", "dents"]

    def __init__(self, mask: Mask, bbox: BoundingBox, conf: float, name: str) -> None:
        super().__init__(mask, bbox, conf, name)
        if name == 0: 
            self.name = "BG"
        elif name == 1: 
            self.name = "Scratches"
        elif name == 2: 
            self.name = "Dents"
        else: 
            self.name = None

class DamageSegmentationModel(MrcnnModel[DamageSegmentationPrediction]):
    def __init__(self, path: str = "./best.pt", **kwargs) -> None:
        super().__init__(path=path, **kwargs)
    
    def _mapIdToName(self, obj: MrcnnPrediction) -> DamageSegmentationPrediction:
        return DamageSegmentationPrediction(obj.mask, obj.boundingBox, obj.confidence, obj.name)

    
    def predict_batch(self, imgs: List[Image]) -> Tuple[Image, Image, List[List[DamageSegmentationPrediction]]]:
        original_imgs, processed_imgs, predictions = super().predict_batch(imgs)
        predictions = list(map(lambda imgPred: map(lambda objPred: self._mapIdToName(objPred), imgPred), predictions))
        return original_imgs, processed_imgs, predictions

class DamageDetectionPrediction(YoloPrediction):
    """ A Class representing a damage prediction

    Attributes
    ----------
    boundingBox: BoundingBox
        BoundingBox
    confidence: float
        Prediction confidence, ranges from 0. to 1.
    """
    name: Literal["Scratches", "Dent"]

class DamageDetectionModel(YoloModel[DamageDetectionPrediction]):
    def __init__(self, path: str = "./best.pt", **kwargs) -> None:
        super().__init__(path=path, **kwargs)
        self.model.conf = kwargs["conf"] if "conf" in kwargs else 0.3
        self.model.iou = kwargs["iou"]  if "iou" in kwargs else 0.45