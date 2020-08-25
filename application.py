import os
import glob
import time
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from flask_caching import Cache




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

@app.route("/")
def hello():
    return "Hello World!!!!!"

@app.route("/ben")
def hellob():
    return render_template('upload.html')

