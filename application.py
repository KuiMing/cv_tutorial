import os
import glob
import time
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from flask_caching import Cache

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!!!!!"

@app.route("/ben")
def hellob():
    return render_template('upload.html')
