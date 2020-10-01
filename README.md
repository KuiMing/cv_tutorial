# Computer Vision Line Chat Bot

## On Azure

### Resource

- Azure App Service: B1

### App service setting

- TLS/SSL setting: 
    - HTTPS only: on 

### Prepare 

1. Install azure cli

2. Turn on Azure app service

3. Open SSH session in browser

4. Edit `/home/config.json`
```
{
    "line":{
            "line_secret":...,
            "line_token":...
    },
    "azure":{
            "subscription_key":...,
            "endpoint":"https://<your name of Azure Cognitive Services>.cognitiveservices.azure.com/",
            "face_key":...,
            "face_end":"https://<your name of Azure Face Detection>.cognitiveservices.azure.com/"
    },
    "imgur":{
            "client_id":...,
            "client_secret":...,
            "access_token":...,
            "refresh_token":...
    }
}
```
5. Set username and password: `az webapp deployment user set --user-name <usrname> --password <password>`

6. Get git url:
`az webapp deployment source config-local-git --name <app_name> --resource-group <resource_name>`

7. Add remote: 
```
cd cv_tutorial
git remote add azure <your_git_url>
```

8. `git push azure master`

## On Heroku

### Prepare

1. Create New heroku app

2. Install [heroku cli](https://devcenter.heroku.com/articles/heroku-cli)

3. Login: `heroku login`

4. Add heroku remote
```
cd cv_tutorial
heroku git:remote -a <your_heroku_app>
```

5. Set heroku environment variables
```
heroku config:set ENDPOINT=https://<your name of Azure Cognitive Services>.cognitiveservices.azure.com/
heroku config:set SUBSCRIPTION_KEY=...
heroku config:set LINE_SECRET=...
heroku config:set LINE_TOKEN=...
heroku config:set IMGUR_ID=...
heroku config:set IMGUR_SECRET=...
heroku config:set IMGUR_ACCESS=...
heroku config:set IMGUR_REFRESH=...
heroku config:set FACE_KEY=...
heroku config:set FACE_END=...
```

6. `git push herku master`

# Pretrained Model

- Face Recognition
  - [Opencv Haar cascade face detection](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
  - [Facenet keras model](https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_)
- Tracker
  - [Goturn model](https://github.com/spmallick/goturn-files)
- Object Detection
  - [YOLO V2 ~ 4](https://github.com/AlexeyAB/darknet#pre-trained-models)

# Open Images Dataset

- [Open Images Dataset](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv)
  - [Annotations of training data](https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv)
  - [Class Names](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv)
  - [Information of training images](https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv)

# Object Detection

- Only YOLO V2, V3 and V4 are applicable.
- In this case, the pre-trained models were trained for MS COCO dataset. So, the file, [`coco.names`](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.names),  contains 80 class names is required.
- Detect Objects in an image.

  example: 
  ```
  python3 object_detector.py -i cat.jpg -c yolov4-tiny.cfg -w yolov4-tiny.weights
  ```

- Detect Objects in a video file or video stream.

  example: 
  ```
  python3 object_detector.py -v test.mp4 -c yolov4-tiny.cfg -w yolov4-tiny.weights
  ```

- Detect Objects with your webcam.

  example: 
  ```
  python3 object_detector.py -v 0 -c yolov4-tiny.cfg -w yolov4-tiny.weights
  ```

# Face Recognition

## Encoding

- Collect at least one image file for each person who should be recognized.
- Put the images of each person into the folders with each person's name.
- Put these folders into a folder.
- Excute:
  ```
  python3 face_encoder.py -i <your image folder> -e <encoding method> -m <keras model> \
  -x <xml for opencv face detector>
  ```
  - Applicable encoding methods: `dlib`, `facenet` and `opencv`
  - If you choose `facenect`, keras model is necessary.
  - If you choose `opencv`, xml for Haar cascade face detection is necessary.
- Output: 
  - `dlib`: `face_data_dlib.pickle`
  - `facenet`: `face_data_facenet.pickle`
  - `opencv`: `face_data.yml`
## Recognition
- Put the encoding results in the same folder with `face_recognizer.py` and `face_tracker.py`.
- Excute `face_recognizer.py`
  ```
  python3.py face_recognizer.py
  ```
  - Press `m` to flip the image.
  - Press `r` to switch recognition method.
  - Press `x` to increase the tolerance, and press `z` to decrease the tolerance.
  - Press `ESC` to quit.

