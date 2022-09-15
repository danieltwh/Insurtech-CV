# Root directory of the project
from numpy import array
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, render_template
from PIL import Image
from io import BytesIO
import base64, sys, os

from scripts.NewCarSidePrediction import CarSideModel
from scripts.NewDamagePrediction import DamageDetectionModel, DamageSegmentationModel
from scripts.CostPrediction import costEstimate

from webutils import getArrayToPlot, produceImage

# Import Mask RCNN to find local version of library
ROOT_DIR = os.path.abspath("./Mask_RCNN")
sys.path.append(ROOT_DIR)  

from mrcnn.utils import resize_image, resize

HOME_TEMPLATE = 'index.html'
ABOUT_TEMPLATE = 'about.html'

app = Flask(__name__)

carSideModel = CarSideModel("weights/Carside_Yolo.pt")
damageDetectModel = DamageDetectionModel("weights/Damage_Yolo.pt")

@app.route('/')
def home():
    return render_template(HOME_TEMPLATE)

    
def resize_image_array(img_arr):
    image, _, _, _, _ = resize_image(
        img_arr,
        min_dim=800,
        min_scale=0,
        max_dim=1024,
        mode="square")
    return image

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
        image = array(Image.open(BytesIO(base64_decoded)))

        image = resize_image_array(image)

        global carSideModel
        global damageDetectModel

        # # Predict: Damage Segmentation
        # damageSegmentModel = DamageSegmentationModel("weights/Damage_MRCNN.h5")
        # image = resize_image_array(image, damageSegmentModel.cfg)

        # _, _, predictions = damageSegmentModel.predict_single(image)
        # _, _, damageSegments = damageSegmentModel.predict_single(image) # Hack because there's a mutation somewhere... TODO: Find the mutation
        # del damageSegmentModel

        # Predict: Carside Detection
        # carSideModel = CarSideModel("weights/Carside_Yolo.pt")
        _, processed_carside, coords = carSideModel.predict_single(image)
        # del carSideModel

        # # Post processing for MRCNN
        # pred_dict = {"bbox": [], "mask": [], "name": [], "score": []}        
        # for prediction in predictions:
        #     pred_dict["bbox"].append(prediction.getBoundingBox())
        #     pred_dict["mask"].append(prediction.getMask())
        #     pred_dict["name"].append(prediction.getName())
        #     pred_dict["score"].append(prediction.getConfidence())
        # pred_bbox = pred_dict["bbox"]
        # pred_mask = pred_dict["mask"]
        # pred_names = pred_dict["name"]
        # pred_conf = pred_dict["score"]
        # output = getArrayToPlot(processed_carside, pred_bbox, pred_mask, pred_names, scores=pred_conf) # Use 'image' to show Yolo damage prediction here also
        # del predictions
        # del pred_bbox, pred_mask, pred_names

        # Predict: Damage Detection
        # damageDetectModel = DamageDetectionModel("weights/Damage_Yolo.pt")
        _, processed_dmg, coords_dmg = damageDetectModel.predict_single(image)
        # del damageDetectModel

        # # Producing images
        # scale = 0.5
        # h,w = output.shape[:2]
        # image_dtype = output.dtype # Save image type before resizing
        # output = resize(output, (round(h * scale), round(w * scale)), preserve_range=True)
        # output = output.astype(image_dtype) # Convert back to original image type
        # pil_img = Image.fromarray(output) # Convert to PIL image

        # uri = produceImage(pil_img, "image/jpeg")
        yolo_uri = produceImage(Image.fromarray(processed_dmg), "yolo_image/jpeg")

        # # Getting estimated costs
        # total_cost = costEstimate(coords, damageSegments, image)
        yolo_total_cost = costEstimate(coords, coords_dmg, image)

        return render_template(HOME_TEMPLATE, filename=filename, yolo_total_cost=yolo_total_cost, yolo_pred=yolo_uri)
    else:
        return redirect(request.url)


@app.route('/about')
def about():
    return render_template(ABOUT_TEMPLATE)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
