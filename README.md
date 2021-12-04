# Fintech Society ML - Insurtech Computer Vision

## Installation

1. Creating a new virtual environment using conda `conda create -n mlvc python=3.7`

2. Select conda VE, then activate the VE `conda activate mlvc`

3. Enter the Mask_RCNN directory `cd Mask_RCNN`

4. Install the necessary dependencies `pip install -r requirements.txt`

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