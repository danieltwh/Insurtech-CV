from PIL import Image
import numpy as np

from scripts.CostPrediction import costEstimate
from scripts.NewCarSidePrediction import CarSideModel
from scripts.NewDamagePrediction import DamageDetectionModel, DamageSegmentationModel

# Loading Models
carSideModel = CarSideModel("weights/Carside_Yolo.pt")
damageDetectModel = DamageDetectionModel("weights/Damage_Yolo.pt")
damageSegmentModel = DamageSegmentationModel("weights/Damage_MRCNN.h5")

# Loading image for demonstration
img = np.asarray(Image.open('scripts/2.jpg'))

# Detections
original, processed, coords = carSideModel.predict_single(img)
_, _, damageSegments = damageSegmentModel.predict_single(img)
_, _, damagePreds = damageDetectModel.predict_single(img)

# Calculate Costs
total_cost = costEstimate(coords, damageSegments, img)
yolo_total_cost = costEstimate(coords, damagePreds, img)

print(total_cost)
print(yolo_total_cost)