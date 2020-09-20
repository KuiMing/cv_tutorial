import argparse
import cv2
import random
from video_tracker import label_object, label_info

# pylint: disable=maybe-no-member

CLASSNAME = "coco.names"
with open(CLASSNAME, "r") as f:
    NAMES = f.read().rstrip("\n").split("\n")


def detect_image(image, model):
    img = cv2.imread(image)
    detect_object(img, model)
    cv2.namedWindow("YOLO", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("YOLO", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("YOLO", img)
    cv2.waitKey()


def detect_object(frame, model, shrink=1):
    small_frame = cv2.resize(frame, dsize=(0, 0), fx=shrink, fy=shrink)
    classes, confidences, locations = model.detect(
        small_frame, confThreshold=0.1, nmsThreshold=0.4
    )
    if len(classes) > 0:
        result = zip(classes.flatten(), confidences.flatten(), locations)
        for classId, confidence, location in result:
            label = "%s: %.2f" % (NAMES[classId], confidence)
            left, top, width, height = [int(i / shrink) for i in location]
            right = left + width
            bottom = top + height
            label_object(frame, left, top, right, bottom, label)


def detect_video(video, model):
    video_cap = cv2.VideoCapture(video)
    mirror = False
    while True:
        timer = cv2.getTickCount()

        ret_val, frame = video_cap.read()
        if not ret_val:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        detect_object(frame, model, 0.25)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        fps_info = "fps: {}".format(str(int(fps)))
        button_info = "Press ESC to quit"
        label_info(frame, button_info, fps_info)

        cv2.namedWindow("YOLO", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("YOLO", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("YOLO", frame)
        keyboard = cv2.waitKey(1)
        # esc to quit
        if keyboard == 27:
            break
        if chr(keyboard & 255) == "m":
            mirror = not mirror


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image path", type=str)
    parser.add_argument("-v", "--video", help="video path", type=str)
    args = parser.parse_args()

    model = cv2.dnn_DetectionModel("yolov4-tiny.cfg", "yolov4-tiny.weights")
    model.setInputParams(size=(416, 416), scale=1 / 255)

    ## Another way
    # net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    ## Fro Nvidia GPU
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    ## After cuda 7.5, it supports FP16.
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    # model = cv2.dnn_DetectionModel(net)
    # model.setInputSize(416, 416)
    # model.setInputScale(1 / 255)

    if args.image:
        detect_image(args.image, model)
    else:
        try:
            video = int(args.video)
        except:
            video = args.video
        detect_video(video, model)


if __name__ == "__main__":
    main()