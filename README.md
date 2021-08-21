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
