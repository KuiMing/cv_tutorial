import os
import glob
import time
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from flask_caching import Cache
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials



def allowed_file(filename):
    """
    detect if file is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = ""
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
CATCHE = Cache(app, config={'CACHE_TYPE': 'null'})
CATCHE.init_app(app)
with open('key', 'r') as f:
    subscription_key = f.read()
f.close()

with open('endpoint', 'r') as f:
    endpoint = f.read()
f.close()

computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))

@app.route("/hello")
def hello():
    return "Hello World!!!!!"


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Upload file
    """
    if request.method == 'POST':
        clear_files()
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            name = str(time.time_ns())
            path = os.path.join(app.config['UPLOAD_FOLDER'], '{}.jpg'.format(name))
            file.save(path)
            redirect(url_for('uploaded_file', filename=path))
            return redirect(url_for('success', name=name))
    return render_template('upload.html')

@app.route('/success/<name>', methods=['GET', 'POST'])
def success(name):
    """
    Display success
    """
    path = os.path.join(app.config['UPLOAD_FOLDER'], '{}.jpg'.format(name))
    remote_image_url = "https://cvlinebot.azurewebsites.net/uploads/{}".format(
        path)
    description_results = computervision_client.describe_image(
        remote_image_url)
    output = ""
    for caption in description_results.captions:
        output += "'{}' with confidence {:.2f}%".format(
            caption.text, caption.confidence * 100)
    return output

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    create uri of upload image
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def clear_files():
    """
    clear files every 5 mins
    """
    file_list = glob.glob("*.jpg")
    if len(file_list) > 0:
        for i in file_list:
            if int(i.replace(".jpg", "")) < (time.time_ns() - 60e9):
                os.remove(i)


