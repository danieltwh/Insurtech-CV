# Root directory of the project
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
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import json
import datetime
import numpy as np
import pandas as pd
# import skimage.draw
import matplotlib.pyplot as plt
import keras

# # Import Mask RCNN
ROOT_DIR = os.path.abspath("./Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library


# from Notebook.inference import PredictionConfig, model_predict, init_model, load_weights

# # Import Mask RCNN
ROOT_DIR = os.path.abspath("./Mask_RCNN")
sys.path.append(ROOT_DIR)  # To find local version of the library

HOME_TEMPLATE = 'index.html'
ABOUT_TEMPLATE = 'about.html'

app = Flask(__name__)


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
    display_instances(image, pred_bbox, pred_mask, pred_class_id,
                      class_names, scores=r['scores'], title='Predicted')
    return r

# # load the train dataset
# train_set = CustomDataset()
# train_set.load_custom('./dataset_train1', 'train')
# train_set.prepare()
# print('Train: %d' % len(train_set.image_ids))

# # load the test dataset
# test_set = CustomDataset()
# test_set.load_custom('./dataset_train1', 'val')
# test_set.prepare()
# print('Test: %d' % len(test_set.image_ids))


def init_model():
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    return cfg, model


def load_weights(model, path):
    # Loading the COCO weights
    model.load_weights(path, by_name=True)


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
        image_string = base64.b64encode(file.read())
        image = Image.open(BytesIO(image_string))
        print(isinstance(Image.open(image), Image.Image))
        # model.predict(image_string)
        # output = model_predict(image_string,model,cfg)
        # print(output["class_ids"])
        return render_template(HOME_TEMPLATE, filename=filename, pred=output['class_ids'])
    else:
        return redirect(request.url)


@app.route('/about')
def about():
    return render_template(ABOUT_TEMPLATE)


if __name__ == "__main__":
    cfg, model = init_model()
    WEIGHTS_PATH = "Notebook/mask_rcnn_damage_0010.h5"
    load_weights(model, WEIGHTS_PATH)
    app.run(port=5000, debug=True)
