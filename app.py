# Root directory of the project

import matplotlib.pyplot as plt
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, render_template
from PIL import Image
from io import BytesIO
import base64
import sys
import os

from scripts.NewCarSidePrediction import CarSideModel
from scripts.NewDamagePrediction import DamageDetectionModel, DamageSegmentationModel

# Import Mask RCNN to find local version of library
ROOT_DIR = os.path.abspath("./Mask_RCNN")
sys.path.append(ROOT_DIR)  
from mrcnn.visualize import apply_mask, random_colors
from mrcnn.utils import resize_image, resize
from skimage.measure import find_contours
from matplotlib import patches
from matplotlib.patches import Polygon
import tensorflow as tf
import io

HOME_TEMPLATE = 'index.html'
ABOUT_TEMPLATE = 'about.html'

app = Flask(__name__)

# Loading Models
carSideModel = CarSideModel("weights/Carside_Yolo.pt")
damageDetectModel = DamageDetectionModel("weights/Damage_Yolo.pt")
damageSegmentModel = DamageSegmentationModel("weights/Damage_MRCNN.h5")

graph = tf.get_default_graph() 

def getArrayToPlot(image, boxes, masks, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = len(boxes)
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        print(boxes)
        print(masks)
        assert len(boxes) == len(masks)

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig.subplots_adjust(0,0,1,1,0,0)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    ax.axis('off')
    ax.set_title(title)
    ax.margins(0,0)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1 = boxes[i].xmin # ["xmin"]
        x2 = boxes[i].xmax # ["xmax"]
        y1 = boxes[i].ymin # ["ymin"]
        y2 = boxes[i].ymax # ["ymax"]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            score = scores[i] if scores is not None else None
            label = class_names[i]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=30, backgroundcolor="none")

        # Mask
        mask = masks[i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

def resize_image_array(img_arr):
    cfg = damageSegmentModel.cfg
    image, window, scale, padding, crop = resize_image(
        img_arr,
        min_dim=cfg.IMAGE_MIN_DIM,
        min_scale=cfg.IMAGE_MIN_SCALE,
        max_dim=cfg.IMAGE_MAX_DIM,
        mode=cfg.IMAGE_RESIZE_MODE)
    return image

@app.route('/')
def home():
    return render_template(HOME_TEMPLATE)


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        print('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        print('No image selected')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        image_string = base64.b64encode(file.stream.read())
        base64_decoded = base64.b64decode(image_string)
        image = Image.open(BytesIO(base64_decoded))
        image_np = np.array(image)
        image = resize_image_array(image_np)
        
        #Mask_RCNN predict       
        with graph.as_default():
            _, _, predictions = damageSegmentModel.predict_single(image)

        pred_dict = {"bbox": [], "mask": [], "name": [], "score": []}        
        for prediction in predictions:
            pred_dict["bbox"].append(prediction.getBoundingBox())
            pred_dict["mask"].append(prediction.getMask())
            pred_dict["name"].append(prediction.getName())
            pred_dict["score"].append(prediction.getConfidence())

        pred_bbox = pred_dict["bbox"]
        pred_mask = pred_dict["mask"]
        pred_names = pred_dict["name"]
        pred_conf = pred_dict["score"]

        output = getArrayToPlot(image, pred_bbox, pred_mask, pred_names, scores=pred_conf)

        scale = 0.5
        h,w = output.shape[:2]
        image_dtype = output.dtype # Save image type before resizing
        output = resize(output, (round(h * scale), round(w * scale)), preserve_range=True)
        output = output.astype(image_dtype) # Convert back to original image type
        pil_img = Image.fromarray(output) # Convert to PIL image
        rawBytes = io.BytesIO()
        pil_img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
        mime = "image/jpeg"
        uri = "data:%s;base64,%s"%(mime, img_base64)

        # Yolo model predict
        _, _, coords = carSideModel.predict_single(image)

        # Yolo model predict (Damage)
        _, processed_dmg, coords_dmg = damageDetectModel.predict_single(image)

        # Printing coords to show correctness
        print(coords)
        print(coords_dmg)
        yolo_pil_img = Image.fromarray(processed_dmg)
        yolo_rawBytes = io.BytesIO()
        yolo_pil_img.save(yolo_rawBytes, "JPEG")
        yolo_rawBytes.seek(0)
        yolo_img_base64 = base64.b64encode(yolo_rawBytes.getvalue()).decode('ascii')
        yolo_mime = "yolo_image/jpeg"
        yolo_uri = "data:%s;base64,%s"%(yolo_mime, yolo_img_base64)

        # Getting the estimated cost for MaskRCNN
        # total_cost = Cost_Estimate(coords, pred_mask, pred_class_id, image)

        # print(pred_mask)
        # print(pred_class_id)

        # Getting the estimated cost for MaskRCNN
        #total_cost = Cost_Estimate(coords, pred_mask, pred_class_id, image)

        # TODO: total_cost
        return render_template(HOME_TEMPLATE, filename=filename, pred=uri, total_cost = 0, yolo_pred=yolo_uri)
    else:
        return redirect(request.url)


@app.route('/about')
def about():
    return render_template(ABOUT_TEMPLATE)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
