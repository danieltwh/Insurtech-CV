from PIL import Image
from skimage.io import imread
from mrcnn.utils import resize_image
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances
from mrcnn.model import mold_image
from mrcnn.model import load_image_gt
from mrcnn.utils import compute_ap
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
from mrcnn.config import Config
from numpy import mean
from numpy import expand_dims
from numpy import asarray
from numpy import zeros
from xml.etree import ElementTree
from os import listdir
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
# import skimage.draw
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import cv2
import PIL


# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN")
# Import Mask RCNN
sys.path.append(ROOT_DIR)
# evaluate the mask rcnn model on the kangaroo dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.visualize import apply_mask, random_colors, display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import resize_image
from skimage.io import imread
from PIL import Image
from skimage.measure import find_contours
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

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
    return img_arr, pred_class_id

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
    return img_arr, pred_class_id


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


cfg, model = init_model()
# Just for testing getting image as array from path
imagedata = imread('./dataset_train2/train/car1.jpg')
imgdata = imagedata.copy()
# Here imgdata should be the image as array and will be resized to fit the model's image size requirements
image = resize_image_array(imgdata)
COCO_WEIGHTS_PATH = './mask_rcnn_damage_0040.h5'
load_weights(model, COCO_WEIGHTS_PATH)

# Outputs a dictionary which contains the predicted mask, class ids, bounding boxes, and scores
output = model_predict(image, model, cfg)

#img_pil = Image.fromarray(output)
#img_pil.show()

# pred_class_id = output['class_ids']
# pred_mask = output['masks']
# pred_bbox = extract_bboxes(pred_mask)



