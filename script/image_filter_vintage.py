"""
Vintage filter
"""
import argparse
import cv2
import numpy as np


# pylint: disable=maybe-no-member
def main():
    """
    Vintage filter
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image path", type=str)
    args = parser.parse_args()

    size = 200
    while True:
        img = cv2.imread(args.image)

        height, width = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(width, size)
        kernel_y = cv2.getGaussianKernel(height, size)
        kernel = kernel_y * kernel_x.T
        vintage_filter = 255 * kernel / np.linalg.norm(kernel)

        for i in range(3):
            img[:, :, i] = img[:, :, i] * vintage_filter
        cv2.imshow("vintage", img)
        keyboard = cv2.waitKey(1)
        # esc to quit
        if keyboard == 27:
            break
        # press x and z to tune filter size
        if chr(keyboard & 255) == "x":
            size += 10
        if chr(keyboard & 255) == "z":
            size -= 10


if __name__ == "__main__":
    main()
