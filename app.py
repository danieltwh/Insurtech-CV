# Root directory of the project
import os
import sys
import base64
from Notebook.inference import model_predict
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

HOME_TEMPLATE = 'index.html'
ABOUT_TEMPLATE = 'about.html'

app = Flask(__name__)


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
        image_string = base64.b64encode(file.stream.read())
        print(model_predict(image_string))
        return render_template(HOME_TEMPLATE, filename=filename)
    else:
        return redirect(request.url)


@app.route('/about')
def about():
    return render_template(ABOUT_TEMPLATE)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
