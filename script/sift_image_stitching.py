"""
SIFT for image stitching
"""
import cv2
import numpy as np
from sift_feature_matching import resize_and_gray_image, parse_args

# pylint: disable=maybe-no-member

np.random.seed(23)


def get_homography(kpst, kpsq, matches, reproj_thresh):
    """
    Get homgraphy between two images
    """
    point_t = np.float32([kp.pt for kp in kpst])
    point_q = np.float32([kp.pt for kp in kpsq])
    if len(matches) > 4:
        pts_t = np.float32([point_t[m.queryIdx] for m in matches])
        pts_q = np.float32([point_q[m.trainIdx] for m in matches])
        homograph, _ = cv2.findHomography(pts_t, pts_q, cv2.RANSAC, reproj_thresh)
        return homograph
    return None


def main():
    """
    SIFT for image stitching
    """
    args = parse_args()
    train_img, train_gray = resize_and_gray_image(args.train_image)
    query_img, query_gray = resize_and_gray_image(args.query_image)

    sift = cv2.xfeatures2d.SIFT_create()

    kps_t, des_t = sift.detectAndCompute(train_gray, None)
    kps_q, des_q = sift.detectAndCompute(query_gray, None)

    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(des_t, des_q, k=2)

    good_matches = []
    for m_1, m_2 in matches:
        if m_1.distance < 0.75 * m_2.distance:
            good_matches.append(m_1)

    width = train_img.shape[1] + query_img.shape[1]
    height = train_img.shape[0] + query_img.shape[0]
    homograph = get_homography(kps_t, kps_q, good_matches, reproj_thresh=4)

    cv2.imshow("keypoint", np.concatenate([train_img, query_img], axis=1))
    cv2.waitKey()

    result = cv2.warpPerspective(train_img, homograph, (width, height))
    cv2.imshow("keypoint", result)
    cv2.waitKey()

    result[0 : query_img.shape[0], 0 : query_img.shape[1]] = query_img
    cv2.imshow("keypoint", result)
    cv2.waitKey()


if __name__ == "__main__":
    main()
