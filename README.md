# Fintech Society ML - Insurtech Computer Vision

## Table of Contents

- [Fintech Society ML - Insurtech Computer Vision](#fintech-society-ml---insurtech-computer-vision)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training and Testing](#training-and-testing)
    - [Deployment](#deployment)
  - [Annotation](#annotation)
  - [Label guide](#label-guide)
  - [YoloV5](#yolov5)
    - [Requirements](#requirements)
    - [Instructions](#instructions)
  - [Directory Structure](#directory-structure)

## Installation

1. Clone this repo as follows

    ```bash
    git clone <THIS_REPO_SSH/HTTPS> --recurse-submodules
    ```

    This is because this repo implicitly uses other repositories.

    If you have accidentally cloned without the `--recurse-submodules` flag, try the following:

    ```bash
    git submodule init
    git submodule update
    ```

2. Install the requirements by running

    ```bash
    pip install -r requirements.txt
    ```

3. Create a new virtual environment using the given yaml file

    ```bash
    conda env update -n mlcv --file mlcv_env.yaml
    ```

    This yaml includes the basic dependencies required by Mask-RCNN, PyTorch, as well as other dependencies needed.

## Usage

Before proceeding, activate the virtual environment: `conda activate mlcv`

### Training and Testing

### Deployment

## Annotation

- Using VGG Image Annotator to annotate (Polygon Tool)
- HTML file is in "Processing" folder, then dataset_v1 folder
- Load project-v1.json file

## Label guide

- If can knock and no internal is seen e.g. dent then it's light (< $400)
- Anything with respray or dent too much is moderate ($400 - $1000)
- Anything with replace is severe (>$1000)

Scratches and dents

## YoloV5

### Requirements

```txt
json
yaml
bidict
```

### Instructions

- Get the `Yolov5` repo as a submodule

    ```bash
    git clone <REPO NAME>
    git submodule init
    git submodule update
    ```

- Generate Yolo-formatted annotations using the `Notebook/Yolov5.ipynb` [Note the configs]
- Edit the `data.yaml` (saved to output_dir) with the corresponding directories. The `data.yaml` file should look like this

    ```yaml
    # All paths are relative to yolov5 directory
    train: data/YoloDataset/train/images 
    val: data/YoloDataset/val/images
    test: data/YoloDataset/test/images # Optional

    # The following is an example
    nc: 3 # Number of Classes
    names: ['Back', 'Front', 'Side'] # Name of classes, sorted.
    ```

- Move the `data.yaml` file to `yolov5/data` directory
- Assuming training and validation data are present in `yolov5/data/YoloDataset` directory, structure workspace as follows:

    ```
    yolov5
        |___data
            |___YoloDataset
                |___train
                    |___images [Put all images here]
                    |___labels [Put all labels here]
                |___val
                    |___images
                    |___labels
                |___test
                    |___images
                    |___labels
            |___data.yaml
    ```

- Run the following command in the yolov5 directory to train:

    ```bash
    python train.py --img 460 --batch 16 --epochs 300 --data data/YoloDataset.yaml --weights yolov5s.pt 
    ```

- For prediction, refer to the [yolov5 github](https://github.com/ultralytics/yolov5). Maybe something like this:

    ```python
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # TODO: load model weights here

    @app.route('/predict', methods = ['POST'])
    def getPrediction(img):
        results = model(img)
        
        # TODO: Format results appropriately
        return results
    ```

## Directory Structure
