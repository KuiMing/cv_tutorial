import cv2
import json
import os
import dlib
from skimage import io
import argparse

# pylint: disable=maybe-no-member

DRAWING = False  # true if mouse is pressed
IX, IY = -1, -1
ANN = []


def label_object(frame, left, top, right, bottom, name):
    cv2.rectangle(
        frame, pt1=(left, top), pt2=(right, bottom), color=(0, 0, 255), thickness=2
    )
    cv2.rectangle(
        frame,
        pt1=(left, bottom - 35),
        pt2=(right, bottom),
        color=(0, 0, 255),
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame,
        text=name,
        org=(left + 6, bottom - 6),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
        thickness=1,
    )


class VideoWorker:
    """
    Split images from video and make Annotations.
    """

    def __init__(self):
        pass

    def split_video(self, source, img_dir):
        """
        Split images from video and save images in specified folder.
        """
        base = os.path.basename(source)
        prefix = os.path.splitext(base)[0]
        # read video
        cam = cv2.VideoCapture(source)
        index = 0
        images = []
        while True:
            ret, frame = cam.read()
            if ret:
                index = index + 1
                index_len10 = str(index).zfill(10)
                frame_name = os.path.abspath(
                    str(img_dir + "/{0}_{1}.jpg".format(prefix, index_len10))
                )
                cv2.imwrite(frame_name, frame)
                images.append(frame_name)
            else:
                cam.release()
                break
        images.sort()
        return images

    def get_all_tags(self, video, ann, video_output, ann_dir, label):
        """
        Track the specified object in first frame and make annotations of others.
        """
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        cam = cv2.VideoCapture(video)
        ret_val, frame = cam.read()
        writer = cv2.VideoWriter(
            video_output, fourcc, 30, (frame.shape[1], frame.shape[0]), True
        )
        tracker = [dlib.correlation_tracker() for _ in range(len(ann))]
        for i, anno in enumerate(ann):
            left = anno["ix"]
            right = anno["x"]
            top = anno["iy"]
            bottom = anno["y"]
            tracker[i].start_track(frame, dlib.rectangle(left, top, right, bottom))
        while True:
            ret_val, frame = cam.read()
            if ret_val:
                for tra in tracker:
                    tra.update(frame)
                    pos = tra.get_position()
                    left = int(pos.left())
                    top = int(pos.top())
                    right = int(pos.right())
                    bottom = int(pos.bottom())
                    label_object(frame, left, top, right, bottom, label)
                    writer.write(frame)
        writer.release()
        cam.release()


def main():

    # mouse callback function
    def draw_rectangle(event, x, y, flags, param):

        global IX, IY, DRAWING, ANN

        # Click
        if event == cv2.EVENT_LBUTTONDOWN:
            DRAWING = True
            IX, IY = x, y
            print(IX, IY)

        elif event == cv2.EVENT_LBUTTONUP:
            DRAWING = False
            cv2.rectangle(img, (IX, IY), (x, y), (0, 255, 255), 0)
            ANN.append({"ix": IX, "iy": IY, "x": x, "y": y})
            print(ANN)

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--video_file_name", help="name of video file", type=str)
    parser.add_argument(
        "-i", "--image_dir", help="path of images", type=str, default="tmp"
    )
    parser.add_argument(
        "-a",
        "--annotation_dir",
        help="path of images",
        type=str,
        default="tmp",
    )
    parser.add_argument("-l", "--label", help="input label", type=str)
    args = parser.parse_args()
    video = args.video_file_name
    cam = cv2.VideoCapture(video)
    _, img = cam.read()
    copyimg = img.copy()
    # img = cv2.imread('room.jpg')
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)

    while True:
        cv2.imshow("image", img)
        keyboard = cv2.waitKey(1) & 0xFF
        if chr(keyboard & 255) == "r":
            cv2.destroyAllWindows()
            img = copyimg
            copyimg = copyimg.copy()
            cv2.namedWindow("image")
            cv2.setMouseCallback("image", draw_rectangle)

        elif chr(keyboard & 255) == "q":
            break
    cam.release()
    cv2.destroyAllWindows()
    print(ANN)

    video_anno = VideoWorker()
    video_anno.get_all_tags(video, ANN, args.image_dir, args.annotation_dir, args.label)


if __name__ == "__main__":
    main()
