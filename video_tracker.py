import sys
import cv2
import dlib
import argparse

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
    def __init__(self, name, threshold):
        self._tracker = dlib.correlation_tracker()
        self._name = name
        self.threshold = threshold

    def init(self, location, frame):
        left, top, width, height = location
        rect = dlib.rectangle(left, top, left + width, top + height)
        self._tracker.start_track(frame, rect)
        label_object(frame, left, top, left + width, top + height, self._name)

    def update(self, frame):
        confidence = self._tracker.update(frame)
        pos = self._tracker.get_position()
        if confidence > self.threshold:
            left = int(pos.left())
            top = int(pos.top())
            right = int(pos.right())
            bottom = int(pos.bottom())
            label_object(frame, left, top, right, bottom, self._name)


class OpencvTracker:
    def __init__(self, name, tracker_name):
        self._tracker = CV_TRACKER[tracker_name]()
        self._name = name

    def init(self, location, frame):
        left, top, width, height = location
        self._tracker.init(frame, location)
        label_object(frame, left, top, left + width, top + height, self._name)

    def update(self, frame):
        ret_val, pos = self._tracker.update(frame)
        if ret_val:
            left = int(pos[0])
            top = int(pos[1])
            right = int(pos[0] + pos[2])
            bottom = int(pos[1] + pos[3])
            label_object(frame, left, top, right, bottom, self._name)


def track_object(frame, bbox, label, tracking):
    if tracking == "dlib":
        track = DlibCorrelationTracker(label, 5)
    else:
        track = OpencvTracker(label, tracking)
    track.init(bbox, frame)
    return track


def label_object(frame, left, top, right, bottom, name):
    height, width, _ = frame.shape
    thick = int((height + width) // 900)
    cv2.rectangle(
        frame, pt1=(left, top), pt2=(right, bottom), color=(0, 0, 255), thickness=thick
    )
    cv2.rectangle(
        frame,
        pt1=(left, bottom - int(35 * 1e-3 * height)),
        pt2=(right, bottom),
        color=(0, 0, 255),
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame,
        text=name,
        org=(left + 6, bottom - 6),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1e-3 * height,
        color=(255, 255, 255),
        thickness=thick,
    )


def label_info(frame, *args):
    height, width, _ = frame.shape
    thick = int((height + width) // 900)
    info = ", ".join(args)
    cv2.putText(
        frame,
        text=info,
        org=(10, 20),
        fontFace=0,
        fontScale=1e-3 * height,
        color=(0, 0, 255),
        thickness=thick,
    )


def main():
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
        label_info(frame, fps_info, tracking_info)

        cv2.imshow("track object", frame)

        keyboard = cv2.waitKey(1)
        # esc to quit
        if keyboard == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
