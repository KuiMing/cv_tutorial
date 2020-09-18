import argparse
import cv2
from PIL import Image, ImageFilter
import numpy as np

from PIL.ImageFilter import (
    BLUR,
    CONTOUR,
    DETAIL,
    EDGE_ENHANCE,
    EDGE_ENHANCE_MORE,
    EMBOSS,
    FIND_EDGES,
    SMOOTH,
    SMOOTH_MORE,
    SHARPEN,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image path", type=str)
    args = parser.parse_args()

    img = cv2.imread(args.image)

    filter_type = [
        "COLORMAP_AUTUMN",
        "COLORMAP_BONE",
        "COLORMAP_JET",
        "COLORMAP_WINTER",
        "COLORMAP_RAINBOW",
        "COLORMAP_OCEAN",
        "COLORMAP_SUMMER",
        "COLORMAP_SPRING",
        "COLORMAP_COOL",
        "COLORMAP_HSV",
        "COLORMAP_PINK",
        "COLORMAP_HOT",
    ]
    for ind, filt in enumerate(filter_type):
        img_filter = cv2.applyColorMap(img, ind)
        cv2.imwrite("{}.jpg".format(filt), img_filter)

    filters = [
        BLUR,
        CONTOUR,
        DETAIL,
        EDGE_ENHANCE,
        EDGE_ENHANCE_MORE,
        EMBOSS,
        FIND_EDGES,
        SMOOTH,
        SMOOTH_MORE,
        SHARPEN,
    ]
    pil_img = Image.fromarray(img)
    for filt in filters:
        img_filter = pil_img.filter(filt)
        img_filter = np.array(img_filter)
        cv2.imwrite("PIL_{}.jpg".format(filt.name.replace(" ", "_")), img_filter)


if __name__ == "__main__":
    main()