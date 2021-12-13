from scripts.NewCarSidePrediction import CarSideModel

import numpy as np
from PIL import Image

# Loading image for demonstration
img = np.asarray(Image.open('scripts/1.jpg'))

# The (only) two lines needed
model = CarSideModel("weights/Carside_Yolo.pt")
original, processed, predictions = model.predict_single(img)

# Printing coords to show correctness
for prediction in predictions:
    print(prediction.getName())
    print(prediction.getConfidence())
    print(prediction.getBoundingBox())

# Saving images to show correctness
original = Image.fromarray(original)
processed = Image.fromarray(processed)

original.save("yolo_original.jpg")
processed.save("yolo_processed.jpg")