from mrcnn.visualize import apply_mask, random_colors
from skimage.measure import find_contours

from matplotlib.pyplot import subplots
from matplotlib import patches
from matplotlib.patches import Polygon

import numpy as np

import io, base64

def produceImage(pil_img, mime="image/jpeg"):
    rawBytes = io.BytesIO()
    pil_img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
    return "data:%s;base64,%s"%(mime, img_base64)

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
        fig, ax = subplots(1, figsize=figsize)
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
        x1 = boxes[i].xmin
        x2 = boxes[i].xmax
        y1 = boxes[i].ymin
        y2 = boxes[i].ymax
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