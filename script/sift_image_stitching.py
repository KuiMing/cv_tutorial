import cv2
import numpy as np
from sift_feature_matching import resize_and_gray_image, parse_args

# pylint: disable=maybe-no-member

np.random.seed(23)


def get_homography(kpsT, kpsQ, matches, reprojThresh):
    point_t = np.float32([kp.pt for kp in kpsT])
    point_q = np.float32([kp.pt for kp in kpsQ])
    if len(matches) > 4:
        pts_t = np.float32([point_t[m.queryIdx] for m in matches])
        pts_q = np.float32([point_q[m.trainIdx] for m in matches])
        homograph, _ = cv2.findHomography(pts_t, pts_q, cv2.RANSAC, reprojThresh)
        return homograph
    else:
        return None


def main():
    args = parse_args()
    train_img, train_gray = resize_and_gray_image(args.train_image)
    query_img, query_gray = resize_and_gray_image(args.query_image)

    sift = cv2.xfeatures2d.SIFT_create()

    kps_t, des_t = sift.detectAndCompute(train_gray, None)
    kps_q, des_q = sift.detectAndCompute(query_gray, None)

    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(des_t, des_q, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    width = train_img.shape[1] + query_img.shape[1]
    height = train_img.shape[0] + query_img.shape[0]
    homograph = get_homography(kps_t, kps_q, good_matches, reprojThresh=4)

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
