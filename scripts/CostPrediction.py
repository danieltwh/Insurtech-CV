import cv2 as cv
import numpy as np
from functools import reduce

from typing import List, Tuple, Union
from scripts.NewCarSidePrediction import CarSidePrediction
from scripts.NewDamagePrediction import DamageDetectionPrediction, DamageSegmentationPrediction
from scripts.predictions.YoloPrediction import YoloPrediction
from scripts.typedefs import BoundingBox, Mask, Image

def bboxToMask(bbox: BoundingBox, height: int, width: int) -> Mask:
    xmin, ymin, xmax, ymax = tuple(map(lambda x: int(np.float32(x)),
                                    (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)))
    return cv.rectangle(np.zeros((height, width)), (xmin, ymin), (xmax, ymax), color=(1, 1, 1), thickness=-1)

def getMaxPrediction(predList: List[YoloPrediction]) -> List[YoloPrediction]:
    '''
    Filters a list of YoloPrediction based on confidence values.
    Only keep the highest probability prediction for each class.
    '''

    results = []
    classNames = list(set(map(lambda pred: pred.getName(), predList)))
    for className in classNames:
        predictions = list(filter(lambda pred: pred.getName() == className, predList))
        maxPred = reduce(lambda currPred, nextPred: currPred if currPred.getConfidence() > nextPred.getConfidence() else nextPred, predictions)
        results.append(maxPred)
    return results


def costEstimate(locList: List[CarSidePrediction], damageList: List[Union[DamageDetectionPrediction, DamageSegmentationPrediction]], image: Image):
    # TODO: Refactor this for easier maintenance
    damage_to_cost = {
        "Scratches" : (200, 400),
        "Dents": (400, 1000),
        "Dent": (400, 1000) # Hack due to model output difference. TODO: Remove this hack
    }

    locList = getMaxPrediction(locList)
    height, width, _ = image.shape
    
    # Get masks
    locationPreds: List[Tuple[YoloPrediction, Mask]] = list(map(lambda location: (location,
                                                                                        bboxToMask(location.getBoundingBox(), height, width))
                                                                , locList))
    damagePreds: List[Tuple[YoloPrediction, Mask]] = list(map(lambda damage: (damage, 
                                                                                        damage.getMask() if hasattr(damage, 'mask')
                                                                                                        else bboxToMask(damage.getBoundingBox(), height, width))
                                                                , damageList))

    # Assigns damage to the appropriate area
    location_with_damages = {}
    for damagePrediction, damageMask in damagePreds:
        locationName = ""
        max_intersect_area = 0

        for locationPrediction, locationMask in locationPreds:
            union = np.count_nonzero(locationMask + damageMask)
            intersect = np.sum(np.where(union==2, 1, 0))
            intersect_area = intersect / union
            if intersect_area > max_intersect_area or (max_intersect_area == 0 and intersect_area == 0):
                locationName = locationPrediction.getName()
                max_intersect_area = intersect_area

        location_with_damages[locationName] = location_with_damages.setdefault(locationName, []) + [damagePrediction]

    # Compute costs
    total_cost = (0, 0)
    for _, damageList in location_with_damages.items():
        if len(damageList) == 0:
            continue
        temp = max([(0, 0)] + list(map(lambda damage: damage_to_cost[damage.getName()], damageList)))
        total_cost = tuple(map(sum, zip(total_cost, temp)))

    total_cost = f"${total_cost[0]} - ${total_cost[1]}"
    return total_cost