import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="image path", type=str)
args = parser.parse_args()

img = cv2.imread(args.image)
rows, cols = img.shape[:2]

kernel_x = cv2.getGaussianKernel(cols, 200)
kernel_y = cv2.getGaussianKernel(rows, 200)
kernel = kernel_y * kernel_x.T
vintage_filter = 255 * kernel / np.linalg.norm(kernel)
vintage_im = np.copy(img)

for i in range(3):
    vintage_im[:, :, i] = vintage_im[:, :, i] * vintage_filter

cv2.imshow("vintage", vintage_im)
cv2.waitKey()