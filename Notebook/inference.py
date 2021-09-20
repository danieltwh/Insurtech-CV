import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import skimage.draw
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

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
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import resize_image
from skimage.io import imread
from PIL import Image

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

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
    display_instances(image, pred_bbox, pred_mask, pred_class_id, class_names , scores=r['scores'], title='Predicted')
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
imagedata=imread('./dataset_train2/train/car1.jpg')
imgdata = imagedata.copy()
# Here imgdata should be the image as array and will be resized to fit the model's image size requirements
image = resize_image_array(imgdata) 
COCO_WEIGHTS_PATH = './mask_rcnn_damage_0040.h5'
load_weights(model,COCO_WEIGHTS_PATH)

output = model_predict(image,model,cfg) # Outputs a dictionary which contains the predicted mask, class ids, bounding boxes, and scores

pred_class_id = output['class_ids']
pred_mask = output['masks']
pred_bbox = extract_bboxes(pred_mask)

# # evaluate model on training dataset
# train_mAP = evaluate_model(train_set, model, cfg)
# print("Train mAP: %.3f" % train_mAP)

# # evaluate model on test dataset
# test_mAP = evaluate_model(test_set, model, cfg)
# print("Test mAP: %.3f" % test_mAP)

