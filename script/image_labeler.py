"""
Label images with rectangle and text
"""
import cv2

# pylint: disable=maybe-no-member


def label_object(frame, left, top, right, bottom, name):
    """
    Label detected objects on images
    """
    height, width, _ = frame.shape
    thick = int((height + width) // 900)
    cv2.rectangle(
        frame, pt1=(left, top), pt2=(right, bottom), color=(0, 0, 255), thickness=thick
    )
    cv2.rectangle(
        frame,
        pt1=(left, bottom - int(35 * 1e-3 * height)),
        pt2=(right, bottom),
        color=(0, 0, 255),
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame,
        text=name,
        org=(left + 6, bottom - 6),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1e-3 * height,
        color=(255, 255, 255),
        thickness=thick,
    )


def label_info(frame, button_info="", *args):
    """
    Put video information on display
    """
    height, width, _ = frame.shape
    thick = int((height + width) // 900)
    info = ", ".join(args)
    cv2.putText(
        frame,
        text=info,
        org=(10, 45),
        fontFace=0,
        fontScale=1e-3 * height,
        color=(0, 0, 255),
        thickness=thick,
    )
    cv2.putText(
        frame,
        text=button_info,
        org=(10, 20),
        fontFace=0,
        fontScale=1e-3 * height,
        color=(0, 0, 200),
        thickness=thick,
    )
