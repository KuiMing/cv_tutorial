import pickle
import cv2
import face_recognition
import dlib

# pylint: disable=maybe-no-member
with open("face_data.pickle", "rb") as f_r:
    NAMES, ENCODINGS = pickle.load(f_r)
f_r.close()

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
        top, right, bottom, left = location
        rect = dlib.rectangle(left, top, right, bottom)
        self._tracker.start_track(frame, rect)
        label_face(frame, left, top, right, bottom, self._name)

    def update(self, frame):
        confidence = self._tracker.update(frame)
        pos = self._tracker.get_position()
        if confidence > self.threshold:
            left = int(pos.left())
            top = int(pos.top())
            right = int(pos.right())
            bottom = int(pos.bottom())
            label_face(frame, left, top, right, bottom, self._name)


class OpencvTracker:
    def __init__(self, name, tracker_name):
        self._tracker = CV_TRACKER[tracker_name]()
        self._name = name

    def init(self, location, frame):
        top, right, bottom, left = location
        rect = (left, top, right - left, bottom - top)
        self._tracker.init(frame, rect)
        label_face(frame, left, top, right, bottom, self._name)

    def update(self, frame):
        ret_val, pos = self._tracker.update(frame)
        if ret_val:
            left = int(pos[0])
            top = int(pos[1])
            right = int(pos[0] + pos[2])
            bottom = int(pos[1] + pos[3])
            label_face(frame, left, top, right, bottom, self._name)


def recognize_track_face(frame, tolerance, tracking):
    shrink = 0.25
    face_locations, face_encodings = detect_face(frame, shrink)
    tracks = []
    for location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = [int(i / shrink) for i in location]
        distances = face_recognition.face_distance(ENCODINGS, face_encoding)
        if min(distances) < tolerance:
            name = NAMES[distances.argmin()]
        else:
            name = "Unknown"
        if tracking == "dlib":
            track = DlibCorrelationTracker(name, 5)
        else:
            track = OpencvTracker(name, tracking)
        track.init((top, right, bottom, left), frame)
        tracks.append(track)
    return tracks


def detect_face(frame, shrink):
    small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    return face_locations, face_encodings


def recognize_face(frame, tolerance):
    shrink = 0.25
    face_locations, face_encodings = detect_face(frame, shrink)
    for location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = [int(i / shrink) for i in location]
        distances = face_recognition.face_distance(ENCODINGS, face_encoding)
        if min(distances) < tolerance:
            name = NAMES[distances.argmin()]
        else:
            name = "Unknown"
        label_face(frame, left, top, right, bottom, name)


def label_face(frame, left, top, right, bottom, name):
    cv2.rectangle(
        frame, pt1=(left, top), pt2=(right, bottom), color=(0, 0, 255), thickness=2
    )
    cv2.rectangle(
        frame,
        pt1=(left, bottom - 35),
        pt2=(right, bottom),
        color=(0, 0, 255),
        thickness=cv2.FILLED,
    )
    cv2.putText(
        frame,
        text=name,
        org=(left + 6, bottom - 6),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=1.0,
        color=(255, 255, 255),
        thickness=1,
    )


def show_face():
    cam = cv2.VideoCapture(0)
    tolerance = 0.5
    mirror = True
    counter = 0
    miss = 0
    tracker_type = [False, "dlib"]
    tracker_type.extend(list(CV_TRACKER.keys()))
    switch = 0
    tracking = False
    while True:
        ret_val, frame = cam.read()

        if not ret_val:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        counter += 1
        counter = counter % 10000
        height, width, _ = frame.shape
        thick = int((height + width) // 900)

        timer = cv2.getTickCount()

        shrink = 0.25
        if counter % 15 == 1:
            small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
            face_locations = face_recognition.face_locations(small_frame)
            if len(face_locations) != len(TRACKERS):
                miss += 1

        if (len(face_locations) != len(TRACKERS)) and (miss >= 10):
            TRACKERS.clear()
            miss = 0

        if tracking:
            if (len(TRACKERS) == 0) and (len(face_locations) > 0):
                track_objs = recognize_track_face(frame, tolerance, tracking)
                TRACKERS.extend(track_objs)
            else:
                for track_obj in TRACKERS:
                    track_obj.update(frame)
        else:
            recognize_face(frame, tolerance)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        fps_info = "fps: {}".format(str(int(fps)))
        tolerance_info = "tolerance: {:.2f}".format(tolerance)
        tracking_info = "Tracker: {}".format(tracking)
        info = ", ".join([fps_info, tolerance_info, tracking_info])
        cv2.putText(
            frame,
            text=info,
            org=(10, 45),
            fontFace=0,
            fontScale=1e-3 * height,
            color=(0, 0, 255),
            thickness=thick,
        )
        cv2.putText(
            frame,
            text="Press t to switch tracker type. Press m to flip image. Presss ESC to quit.",
            org=(10, 20),
            fontFace=0,
            fontScale=1e-3 * height,
            color=(0, 0, 200),
            thickness=thick,
        )
        cv2.imshow("track face", frame)

        keyboard = cv2.waitKey(1)
        # esc to quit
        if keyboard == 27:
            break
        # press m to flip frame
        if chr(keyboard & 255) == "m":
            mirror = not mirror
        # switch tracker
        if chr(keyboard & 255) == "t":
            TRACKERS.clear()
            switch += 1
            switch %= len(tracker_type)
            tracking = tracker_type[switch]
    cam.release()
    cv2.destroyAllWindows()


def main():
    show_face()


if __name__ == "__main__":
    print("Press t to switch tracker type. Press m to flip image. Presss ESC to quit.")
    main()
