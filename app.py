# Root directory of the project
from mrcnn.model import MaskRCNN
import os
import sys
import base64
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
# Import Mask RCNN
ROOT_DIR = os.path.abspath("./Mask_RCNN")

sys.path.append(ROOT_DIR)  # To find local version of the library

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('testing.html')


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
        print('upload_image filename: ' + filename)
        return render_template('testing.html', filename=filename)
    else:
        return redirect(request.url)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
