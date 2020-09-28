"""
Extract keypoints of images with SIFT
"""
import argparse
import cv2
import numpy as np
import imageio

# pylint: disable=maybe-no-member
np.random.seed(23)


def main():
    """
    Extract keypoints of images with SIFT
    """
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
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except FileNotFoundError:
        img = cv2.imread(args.image)

    img = cv2.resize(img, None, fx=0.1, fy=0.1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    keypoints = sift.detect(gray, None)
    img_sift = img.copy()

    img_sift = cv2.drawKeypoints(
        img, keypoints, img_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imshow("keypoint", np.concatenate([img, img_sift], axis=1))
    cv2.waitKey()


if __name__ == "__main__":
    main()
