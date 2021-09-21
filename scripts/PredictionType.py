from typing import TypedDict

class DetectionPrediction(TypedDict):
    # (0, 0) is top-left of the image
    # bottom-right of the image is most positive coordinates
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    confidence: float   # Prediction confidence: [0., 1.]
    name: str           # Prediction name