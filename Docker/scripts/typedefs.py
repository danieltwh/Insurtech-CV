from abc import abstractmethod, ABCMeta
from typing import TypeVar, Sequence, Generic, List, Tuple
import numpy.typing as npt

Image = npt.ArrayLike

T = TypeVar('T')
npArray = Sequence[T] 

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

T = TypeVar("T", bound=DetectionPrediction)
class Model(Generic[T], metaclass=ABCMeta):
    """ Parent class of all models
    """
    @abstractmethod
    def predict_batch(self, imgs:List[Image]) -> Tuple[npArray[Image], npArray[Image], List[T]]:
        pass

    @abstractmethod
    def predict_single(self, img:Image) -> Tuple[Image, Image, List[T]]:
        pass