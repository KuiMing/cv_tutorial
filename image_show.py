import argparse
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image path", type=str)

    args = parser.parse_args()

    img = cv2.imread(args.image)
    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("image", 1000, 600)
    cv2.imshow("image", img)
    cv2.moveWindow("image", 0, 0)
    cv2.waitKey()


if __name__ == "__main__":
    main()
