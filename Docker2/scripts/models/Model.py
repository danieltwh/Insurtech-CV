from abc import abstractmethod, ABCMeta
from typing import TypeVar, Generic, List, Tuple

from scripts.typedefs import Image, DetectionPrediction, npArray

T = TypeVar("T", bound=DetectionPrediction)
class Model(Generic[T], metaclass=ABCMeta):
    """ Parent class of all models

    The website makes use of these methods to format prediction. Thus, any model to be used
    with the website must inherit from this class. 
    """
    
    @abstractmethod
    def predict_batch(self, imgs:List[Image]) -> Tuple[npArray[Image], npArray[Image], List[T]]:
        pass

    @abstractmethod
    def predict_single(self, img:Image) -> Tuple[Image, Image, List[T]]:
        pass