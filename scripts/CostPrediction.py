import cv2 as cv
import numpy as np

from typing import List
from scripts.predictions.YoloPrediction import YoloPrediction
from scripts.predictions.MrcnnPrediction import MrcnnPrediction
from scripts.typedefs import Mask, Image

def Cost_Estimate(loc_coords: List[MrcnnPrediction], damage_mask: List[Mask], pred_names: List[str], image: Image):
    location_mask = {}
    location_confidence = {}
    height, width, _ = image.shape
    for loc in loc_coords:
        curr_name = loc.getName()
        curr_confidence = loc.getConfidence()

        if curr_name not in location_confidence or location_confidence.get(curr_name) < curr_confidence:
            mask = np.zeros((height, width))

            bbox = loc.getBoundingBox()
            xmin, ymin, xmax, ymax = tuple(map(lambda x: int(np.float32(x)),(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)))

            mask = cv.rectangle(mask, (xmin, ymin), (xmax, ymax), color=(1, 1, 1), thickness=-1)

            location_mask[curr_name] = mask
            location_confidence[curr_name] = curr_confidence

    location_with_damages = {} 

    for key, val in location_mask.items():
        location_with_damages[key] = []

    for i, damage_name in enumerate(pred_names):
        mask = np.copy(damage_mask[i])

        curr_area = np.count_nonzero(mask)
        # print(damage_name, curr_area)

        damage_loc = ""
        max_intersect_area = 0

        for location, mask2 in location_mask.items():
            combined = mask + mask2
            intersect = np.sum(np.where(combined==2, 1, 0))
            intersect_area = intersect / curr_area
            if intersect_area > max_intersect_area or (max_intersect_area == 0 and intersect_area == 0):
                damage_loc = location
                max_intersect_area = intersect_area

        location_with_damages[damage_loc].append(damage_name)

    damage_to_cost = {
        "Scratches" : (200, 400),
        "Dents": (400, 1000)
    }

    total_cost = (0, 0)

    for key, val in location_with_damages.items():
        if len(val) == 0:
            break
        temp = max(list(map(lambda x: damage_to_cost[x], val)))
        total_cost = tuple(map(sum, zip(total_cost, temp)))

    total_cost = f"${total_cost[0]} - ${total_cost[1]}"
    return total_cost

# yet to edit
def Cost_Estimate_YOLO(loc_coords: List[YoloPrediction], dmg_coords: List[YoloPrediction], pred_names: List[str], image: Image):
    location_mask = {}
    location_confidence = {}
    height, width, _ = image.shape
    for loc in loc_coords:
        curr_name = loc.getName()
        curr_confidence = loc.getConfidence()

        if curr_name not in location_confidence or location_confidence.get(curr_name) < curr_confidence:
            mask = np.zeros((height, width))
            bbox = loc.getBoundingBox()
            xmin, ymin, xmax, ymax = tuple(map(lambda x: int(np.float32(x)),(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)))

            mask = cv.rectangle(mask, (xmin, ymin), (xmax, ymax), color=(1, 1, 1), thickness=-1)

            location_mask[curr_name] = mask
            location_confidence[curr_name] = curr_confidence
    
    damage_mask = {}
    for loc in dmg_coords:
        curr_name = loc.getName()
        curr_confidence = loc.getConfidence()

        if curr_name not in location_confidence or location_confidence.get(curr_name) < curr_confidence:
            mask = np.zeros((height, width))
            bbox = loc.getBoundingBox()
            xmin, ymin, xmax, ymax = tuple(map(lambda x: int(np.float32(x)),(bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax)))

            mask = cv.rectangle(mask, (xmin, ymin), (xmax, ymax), color=(1, 1, 1), thickness=-1)

            damage_mask[curr_name] = mask

    location_with_damages = {} 

    for key, val in location_mask.items():
        location_with_damages[key] = []

    for i, damage_name in enumerate(pred_names):
        mask = np.copy(damage_mask[damage_name])

        curr_area = np.count_nonzero(mask)

        damage_loc = ""
        max_intersect_area = 0

        for location, mask2 in location_mask.items():
            combined = mask + mask2
            intersect = np.sum(np.where(combined==2, 1, 0))
            intersect_area = intersect / curr_area
            # print(location, intersect_area)
            if intersect_area > max_intersect_area or (max_intersect_area == 0 and intersect_area == 0):
                damage_loc = location
                max_intersect_area = intersect_area

        location_with_damages[damage_loc].append(damage_name)

    # print(location_with_damages)

    damage_to_cost = {
        "Scratches" : (200, 400),
        "Dents": (400, 1000)
    }

    total_cost = (0, 0)

    for key, val in location_with_damages.items():
        if len(val) == 0:
            break
        temp = max(list(map(lambda x: damage_to_cost[x], val)))
        total_cost = tuple(map(sum, zip(total_cost, temp)))

    total_cost = f"${total_cost[0]} - ${total_cost[1]}"
    return total_cost

if __name__ == "__main__":
    from CarSidePrediction import YoloModel
    # from DamagePrediction import *
    import sys
    import os
    ROOT_DIR = os.path.abspath("../Mask_RCNN")
    sys.path.append(ROOT_DIR)  
    # To find local version of the library
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy import expand_dims
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
    cfg, MRCNNModel = init_model()
    COCO_WEIGHTS_PATH = '../Notebook/mask_rcnn_damage_0010.h5'
    load_weights(MRCNNModel, COCO_WEIGHTS_PATH)

    # Loading image for demonstration
    from PIL import Image
    img = np.asarray(Image.open('1.jpg'))

    # Location Detection
    loc_model = YoloModel("./best.pt")
    original, processed, coords = loc_model.predict_single(img)
    
    # Damage Detection
    output,pred_class_id, damage_mask = model_predict(img , MRCNNModel, cfg)

    total_cost = cost_estimate(coords, damage_mask, pred_class_id, img)
    print(total_cost)