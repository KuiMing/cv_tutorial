"""
Track selected objects on the video
"""
import argparse
import sys
import cv2
import dlib
from image_labeler import label_object, label_info

# pylint: disable=maybe-no-member

TRACKERS = []
CV_TRACKER = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "mosse": cv2.TrackerMOSSE_create,
    "boosting": cv2.TrackerBoosting_create,
    "tld": cv2.TrackerTLD_create,
    "goturn": cv2.TrackerGOTURN_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mil": cv2.TrackerMIL_create,
}


class DlibCorrelationTracker:
    """
    Dlib tracker
    """

    def __init__(self, name, threshold):
        self._tracker = dlib.correlation_tracker()
        self._name = name
        self.threshold = threshold

    def init(self, location, frame):
        """
        Initial tracker
        """
        left, top, width, height = location
        rect = dlib.rectangle(left, top, left + width, top + height)
        self._tracker.start_track(frame, rect)
        label_object(frame, left, top, left + width, top + height, self._name)

    def update(self, frame):
        """
        Update tracker
        """
        confidence = self._tracker.update(frame)
        pos = self._tracker.get_position()
        if confidence > self.threshold:
            left = int(pos.left())
            top = int(pos.top())
            right = int(pos.right())
            bottom = int(pos.bottom())
            label_object(frame, left, top, right, bottom, self._name)


class OpencvTracker:
    """
    Opencv trackers
    """

    def __init__(self, name, tracker_name):
        self._tracker = CV_TRACKER[tracker_name]()
        self._name = name

    def init(self, location, frame):
        """
        Initial tracker
        """
        left, top, width, height = location
        self._tracker.init(frame, location)
        label_object(frame, left, top, left + width, top + height, self._name)

    def update(self, frame):
        """
        Update tracker
        """
        ret_val, pos = self._tracker.update(frame)
        if ret_val:
            left = int(pos[0])
            top = int(pos[1])
            right = int(pos[0] + pos[2])
            bottom = int(pos[1] + pos[3])
            label_object(frame, left, top, right, bottom, self._name)


def track_object(frame, bbox, label, tracking):
    """
    track the object
    """
    if tracking == "dlib":
        track = DlibCorrelationTracker(label, 5)
    else:
        track = OpencvTracker(label, tracking)
    track.init(bbox, frame)
    return track


def main():
    """
    Select a object and track it.
    """
    tracker_type = list(CV_TRACKER.keys())
    tracker_type.append("dlib")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="video path", type=str)
    parser.add_argument(
        "-t", "--tracking", help=", ".join(tracker_type), type=str, default="dlib"
    )
    parser.add_argument("-l", "--label", help="input label", type=str)
    args = parser.parse_args()
    video = args.input
    tracking = args.tracking
    label = args.label
    video_cap = cv2.VideoCapture(video)
    if not video_cap.isOpened():
        print("Could not open video")
        sys.exit()

    ret_val, frame = video_cap.read()
    if not ret_val:
        print("Cannot read video file")
        sys.exit()

    bbox = cv2.selectROI(frame, False)
    track_obj = track_object(frame, bbox, label, tracking)

    while True:
        timer = cv2.getTickCount()

        ret_val, frame = video_cap.read()
        if not ret_val:
            break

        track_obj.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        fps_info = "fps: {}".format(str(int(fps)))
        tracking_info = "Tracker: {}".format(tracking)
        button_info = "Press ESC to quit"
        label_info(frame, button_info, fps_info, tracking_info)

        cv2.namedWindow("Track", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Track", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Track", frame)

        keyboard = cv2.waitKey(1)
        # esc to quit
        if keyboard == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
