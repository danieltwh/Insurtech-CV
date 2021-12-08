# Fintech Society ML - Insurtech Computer Vision

## Table of Contents

- [Fintech Society ML - Insurtech Computer Vision](#fintech-society-ml---insurtech-computer-vision)
  - [Table of Contents](#table-of-contents)
  - [Notes](#notes)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training and Testing](#training-and-testing)
    - [Deployment](#deployment)
  - [Annotation](#annotation)
  - [Label guide](#label-guide)
  - [Miscellaneous](#miscellaneous)
    - [Yolov5 Training / Inference](#yolov5-training--inference)

## Notes

1. This repository makes use of Matterport's implementation of Mask-RCNN:
    - Tensorflow 1.15
    - Supports Python <=3.7 (Tested on Python3.7)

    Please use `virtualenv` to create an environment with python3.7. Other python versions may lead to installation issues

2. By default, the provided `requirements.txt` provides installation for CPU version of the libraries i.e. Tensorflow and PyTorch. Please install the GPU versions if necessary

```bash
python3 -m pip install tensorflow-gpu=1.15 cudatoolkit
```

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
    python3 -m pip install -r requirements.txt
    ```

## Usage

### Training and Testing

Jupyter notebooks are provided in the `development` folder.

### Deployment

The website is deployed using `flask`. Code for this is in `app.py`, and the scripts for ML related services are in `scripts/` folder.

## Annotation

- Using VGG Image Annotator to annotate (Polygon Tool)
- HTML file is in "Processing" folder, then dataset_v1 folder
- Load project-v1.json file

## Label guide

- If can knock and no internal is seen e.g. dent then it's light (< $400)
- Anything with respray or dent too much is moderate ($400 - $1000)
- Anything with replace is severe (>$1000)

Scratches and dents

## Miscellaneous

### Yolov5 Training / Inference

- Generate Yolo-formatted annotations using the `development/Notebook/Yolov5.ipynb` [Note the configs]
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

    ```bash
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
