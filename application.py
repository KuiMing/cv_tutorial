import os
import glob
import time
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory, abort

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import json
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage,
                            FlexSendMessage, ImageMessage, ImageSendMessage)
from imgur_python import Imgur
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = ""

with open('/home/config.json', 'r') as f:
    config = json.load(f)
f.close()

subscription_key = config['azure']['subscription_key']
endpoint = config['azure']['endpoint']
computervision_client = ComputerVisionClient(
    endpoint, CognitiveServicesCredentials(subscription_key))

line_secret = config['line']['line_secret']
line_token = config['line']['line_token']
line_bot_api = LineBotApi(line_token)
handler = WebhookHandler(line_secret)

imgur_config = config['imgur']
imgur_client = Imgur(config=imgur_config)


def azure_describe(remote_image_url):
    description_results = computervision_client.describe_image(
        remote_image_url)
    output = ""
    for caption in description_results.captions:
        output += "'{}' with confidence {:.2f}% \n".format(
            caption.text, caption.confidence * 100)
    return output


def azure_object_detection(url, filename):
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(
        "static/TaipeiSansTCBeta-Regular.ttf", size=int(5e-2 * img.size[1]))
    object_detection = computervision_client.detect_objects(url)
    if len(object_detection.objects) > 0:
        for obj in object_detection.objects:
            left = obj.rectangle.x
            top = obj.rectangle.y
            right = obj.rectangle.x + obj.rectangle.w
            bot = obj.rectangle.y + obj.rectangle.h
            name = obj.object_property
            confidence = obj.confidence
            print("{} at location {}, {}, {}, {}".format(
                name, left, right, top, bot))
            draw.rectangle(
                [left, top, right, bot], outline=(255, 0, 0), width=3)
            draw.text(
                [left, abs(top - 12)],
                "{} {}".format(name, confidence),
                fill=(255, 0, 0),
                font=fnt)
    img.save(filename)
    image = imgur_client.image_upload(filename, 'first', 'first')
    link = image['response']['data']['link']
    os.remove(filename)
    return link


@app.route("/")
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


@handler.add(MessageEvent, message=ImageMessage)
def handle_content_message(event):
    if isinstance(event.message, ImageMessage):
        print(event.message)
        print(event.source.user_id)
        print(event.message.id)
        filename = "{}.jpg".format(event.message.id)
        message_content = line_bot_api.get_message_content(event.message.id)
        with open(filename, 'wb') as fd:
            for chunk in message_content.iter_content():
                fd.write(chunk)
        image = imgur_client.image_upload(filename, 'first', 'first')
        link = image['response']['data']['link']
        output = azure_describe(link)
        link = azure_object_detection(link, filename)
        bubble = {
            "type": "bubble",
            "header": {
                "type":
                "box",
                "layout":
                "vertical",
                "contents": [{
                    "type":
                    "box",
                    "layout":
                    "horizontal",
                    "contents": [{
                        "type":
                        "box",
                        "layout":
                        "vertical",
                        "contents": [{
                            "type": "image",
                            "url": link,
                            "size": "full",
                            "aspectMode": "cover",
                            "aspectRatio": "1:1",
                            "gravity": "center"
                        }],
                        "flex":
                        1
                    }]
                }],
                "paddingAll":
                "0px"
            },
            "body": {
                "type":
                "box",
                "layout":
                "vertical",
                "contents": [{
                    "type":
                    "box",
                    "layout":
                    "vertical",
                    "contents": [{
                        "type":
                        "box",
                        "layout":
                        "vertical",
                        "contents": [{
                            "type": "text",
                            "text": output,
                            "color": "#ffffffcc",
                            "size": "sm",
                            "wrap": True
                        }],
                        "spacing":
                        "sm"
                    }],
                    "height":
                    "50px"
                }],
                "paddingAll":
                "20px",
                "backgroundColor":
                "#464F69"
            }
        }
        line_bot_api.reply_message(
            event.reply_token,
            [FlexSendMessage(alt_text="Report", contents=bubble)])
