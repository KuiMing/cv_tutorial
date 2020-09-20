import argparse
import cv2
import random
from video_tracker import label_object, label_info

# pylint: disable=maybe-no-member


## Another way
# net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
## Fro Nvidia GPU
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
## After cuda 7.5, it supports FP16.
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
# model = cv2.dnn_DetectionModel(net)
# model.setInputSize(608, 608)
# model.setInputScale(1 / 255)


def detect_image(image, model, names):
    img = cv2.imread(image)
    classes, confidences, boxes = model.detect(img, confThreshold=0.1, nmsThreshold=0.4)
    if len(classes) > 0:
        for classId, confidence, box in zip(
            classes.flatten(), confidences.flatten(), boxes
        ):
            label = "%s: %.2f" % (names[classId], confidence)
            left, top, width, height = box
            right = left + width
            bottom = top + height
            label_object(img, left, top, right, bottom, label)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("image", img)
    cv2.waitKey()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image path", type=str)
    parser.add_argument("-v", "--video", help="video path", type=str)
    args = parser.parse_args()
    className = "coco.names"
    with open(className, "r") as f:
        names = f.read().rstrip("\n").split("\n")

    model = cv2.dnn_DetectionModel("yolov4-tiny.cfg", "yolov4-tiny.weights")
    model.setInputParams(size=(416, 416), scale=1 / 255)
    detect_image(args.image, model, names)


# video_cap = cv2.VideoCapture(0)

# while True:
#     ret_val, frame = video_cap.read()
#     if ret_val:
#         shrink = 0.25
#         small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
#         classes, confidences, boxes = model.detect(
#             small_frame, confThreshold=0.1, nmsThreshold=0.4
#         )
#         if len(classes) > 0:

#             for classId, confidence, location in zip(
#                 classes.flatten(), confidences.flatten(), boxes
#             ):
#                 label = "%s: %.2f" % (names[classId], confidence)
#                 left, top, width, height = [int(i / shrink) for i in location]
#                 right = left + width
#                 bottom = top + height
#                 label_object(frame, left, top, right, bottom, label)

#     cv2.imshow("YOLO v4", frame)

#     keyboard = cv2.waitKey(1)
#     # esc to quit
#     if keyboard == 27:
#         break


if __name__ == "__main__":
    main()
