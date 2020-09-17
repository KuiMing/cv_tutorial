import argparse
import cv2
from PIL import Image, ImageFilter
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image path", type=str)
    args = parser.parse_args()

    img = cv2.imread(args.image)

    img_filter = cv2.applyColorMap(img, 10)

    # pil_img = Image.fromarray(img)
    # pil_img = pil_img.filter(ImageFilter.FIND_EDGES)
    # img_filter = np.array(pil_img)

    cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("image", img_filter)
    cv2.moveWindow("image", 0, 0)
    cv2.waitKey()


if __name__ == "__main__":
    main()
