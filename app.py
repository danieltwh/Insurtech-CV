# Root directory of the project

import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import json
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from PIL import Image
from io import BytesIO
import base64
import sys
import os
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean

from scripts.CarSidePrediction import YoloModel
from scripts.CostPrediction import Cost_Estimate
# Import Mask RCNN
ROOT_DIR = os.path.abspath("./Mask_RCNN")
sys.path.append(ROOT_DIR)  
# To find local version of the library
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.visualize import apply_mask, random_colors, display_instances
from mrcnn.utils import extract_bboxes, resize_image, resize
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import tensorflow as tf
import io
import cv2


# import skimage.draw


# from Notebook.inference import PredictionConfig, model_predict, init_model, load_weights



HOME_TEMPLATE = 'index.html'
ABOUT_TEMPLATE = 'about.html'

app = Flask(__name__)


def get_array_from_plot(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)
        fig.subplots_adjust(0,0,1,1,0,0)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    #ax.set_ylim(height + 10, -10)
    #ax.set_xlim(-10, width + 10)
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
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=30, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
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

# define the prediction configuration


class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "damage_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 2
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def model_predict(image, model, cfg):
    class_names = ['BG', 'scratches', 'dents']
    # load image, bounding boxes and masks for the image id
    # image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)
    # extract results for first sample
    r = yhat[0]
    # Getting predicted values from r
    pred_class_id = r['class_ids']
    pred_mask = r['masks']
    pred_bbox = extract_bboxes(pred_mask)
    # display predicted image with masks and bounding boxes
    img_arr = get_array_from_plot(image, pred_bbox, pred_mask, pred_class_id, class_names , scores=r['scores'], title='Predicted')
    return img_arr, pred_class_id, pred_mask


def init_model():
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    return cfg, model


def load_weights(model, path):
    # Loading the COCO weights
    model.load_weights(path, by_name=True)


def resize_image_array(img_arr):
    image, window, scale, padding, crop = resize_image(
        img_arr,
        min_dim=cfg.IMAGE_MIN_DIM,
        min_scale=cfg.IMAGE_MIN_SCALE,
        max_dim=cfg.IMAGE_MAX_DIM,
        mode=cfg.IMAGE_RESIZE_MODE)
    return image

graph = tf.get_default_graph()
cfg, model = init_model()
COCO_WEIGHTS_PATH = './Notebook/mask_rcnn_damage_0010.h5'
load_weights(model, COCO_WEIGHTS_PATH)

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
        # image_string = file.read()
        # image_np = np.array(np.fromstring(image_string, np.uint8))
        image_np = np.array(image)
        image = resize_image_array(image_np)

        # Yolo model predict
        yolo_model = YoloModel("./scripts/best.pt")
        original, processed, coords = yolo_model.predict_single(image)
        # Printing coords to show correctness
        print(coords)
        yolo_pil_img = Image.fromarray(processed)
        yolo_rawBytes = io.BytesIO()
        yolo_pil_img.save(yolo_rawBytes, "JPEG")
        yolo_rawBytes.seek(0)
        yolo_img_base64 = base64.b64encode(yolo_rawBytes.getvalue()).decode('ascii')
        mime = "image/jpeg"
        yolo_uri = "data:%s;base64,%s"%(mime, yolo_img_base64)

        #Mask_RCNN predict
        with graph.as_default():
            output,pred_class_id, pred_mask = model_predict(image ,model, cfg)

        # Getting the estimated cost
        total_cost = Cost_Estimate(coords, pred_mask, pred_class_id, image)

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

        return render_template(HOME_TEMPLATE, filename=filename, pred=uri, total_cost = total_cost)
    else:
        return redirect(request.url)


@app.route('/about')
def about():
    return render_template(ABOUT_TEMPLATE)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
