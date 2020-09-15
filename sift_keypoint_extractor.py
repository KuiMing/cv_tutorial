import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse

np.random.seed(23)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image",
        help="image path or url",
        type=str,
        default="https://i.imgur.com/8Imc4ax.jpg",
    )

    args = parser.parse_args()

    try:
        img = imageio.imread(args.image)
    except FileNotFoundError:
        img = cv2.imread(args.image)

    img = cv2.resize(img, None, fx=0.1, fy=0.1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(gray, None)
    img_sift = img.copy()

    img_sift = cv2.drawKeypoints(
        img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.subplot(121)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Raw Image")

    plt.subplot(122)
    img_sift_rgb = cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)
    plt.imshow(img_sift_rgb)
    plt.title("Image with keypoints")
    plt.show()


if __name__ == "__main__":
    main()
