from scripts.NewDamagePrediction import DamageDetectionModel, DamageSegmentationModel

import numpy as np
from PIL import Image

def loadDetect():
    return DamageDetectionModel("weights/Damage_Yolo.pt")

def loadSegment():
    return DamageSegmentationModel("weights/Damage_MRCNN.h5")

# Loading image for demonstration
img = np.asarray(Image.open('scripts/1.jpg'))

# The (only) two lines needed
model = loadSegment()
original, processed, predictions = model.predict_single(img)

# Printing to show correctness
for prediction in predictions:
    print(prediction.getName())
    print(prediction.getConfidence())
    print(prediction.getBoundingBox())
    print(prediction.getMask())

# Saving images to show correctness
original = Image.fromarray(original)
processed = Image.fromarray(processed)

original.save("yolo_original.jpg")
processed.save("yolo_processed.jpg")