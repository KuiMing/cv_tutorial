import os
import glob
import time
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from flask_caching import Cache
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import json
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage,
                            FlexSendMessage)


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
with open('/home/key', 'r') as f:
    subscription_key = f.read()
f.close()
subscription_key = subscription_key.replace('\n', '')

with open('/home/endpoint', 'r') as f:
    endpoint = f.read()
f.close()
endpoint = endpoint.replace('\n', '')

computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))

with open('/home/line_config.json', 'r') as f:
    config = json.load(f)
f.close()
line_secret = config['line_secret']
line_token = config['line_token']
line_bot_api = LineBotApi(line_token)
handler = WebhookHandler(line_secret)


def azure_describe(remote_image_url):
    description_results = computervision_client.describe_image(
        remote_image_url)
    output = ""
    for caption in description_results.captions:
        output += "'{}' with confidence {:.2f}% \n".format(
            caption.text, caption.confidence * 100)
    return output


@app.route("/hello")
def hello():
    return "Hello World!!!!!"


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print(body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print(
            "Invalid signature. Please check your channel access token/channel secret."
        )
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    message = TextSendMessage(text=event.message.text)
    print(event.source.user_id)
    print(event.source.type)
    # print(line_bot_api.get_room_member_ids(room_id))
    line_bot_api.reply_message(event.reply_token, message)

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
            path = os.path.join(app.config['UPLOAD_FOLDER'],
                                '{}.jpg'.format(name))
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
