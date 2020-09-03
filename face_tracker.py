import time
import pickle
import cv2
import face_recognition
import dlib

# pylint: disable=maybe-no-member
with open("face_data.pickle", "rb") as f_r:
    NAMES, ENCODINGS = pickle.load(f_r)
f_r.close()

TRACKERS = []
LOCATIONS = []


class DlibCorrelationTracker():
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
        left = int(pos.left())
        top = int(pos.top())
        right = int(pos.right())
        bottom = int(pos.bottom())
        if confidence > self.threshold:
            label_face(frame, left, top, right, bottom, self._name)


def recognize_track_face(frame, tolerance):
    shrink = 0.25
    face_locations, face_encodings = detect_face(frame, shrink)
    for location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = [int(i / shrink) for i in location]
        distances = face_recognition.face_distance(ENCODINGS, face_encoding)
        if min(distances) < tolerance:
            name = NAMES[distances.argmin()]
        else:
            name = "Unknown"
        track = DlibCorrelationTracker(name, 10)
        track.init((top, right, bottom, left), frame)
        TRACKERS.append(track)

def detect_face(frame, shrink):
    small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame,
                                                     face_locations)
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
    cv2.rectangle(frame,
                  pt1=(left, top),
                  pt2=(right, bottom),
                  color=(0, 0, 255),
                  thickness=2)
    cv2.rectangle(frame,
                  pt1=(left, bottom - 35),
                  pt2=(right, bottom),
                  color=(0, 0, 255),
                  thickness=cv2.FILLED)
    cv2.putText(frame,
                text=name,
                org=(left + 6, bottom - 6),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=1)


def show_face():
    cam = cv2.VideoCapture(0)
    tolerance = 0.5
    mirror = True
    now = time.time()
    counter = 0
    miss = 0
    tracking = True
    while True:
        ret_val, frame = cam.read()
        if mirror:
            frame = cv2.flip(frame, 1)

        if ret_val:
            counter += 1
            counter = counter % 10000
            height, width, _ = frame.shape
            thick = int((height + width) // 900)
            if counter % 5 == 1:
                fps = 5 / (time.time() - now)
                now = time.time()
                fps_info = "fps: {}".format(str(int(fps)).zfill(2))
                shrink = 0.25

            if counter % 15 == 1:
                small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
                face_locations = face_recognition.face_locations(small_frame)
                if len(face_locations) == 0:
                    miss += 1

            if (len(face_locations) == 0) and (miss >= 10):
                TRACKERS.clear()
                miss = 0

            tolerance_info = "tolerance: {:.2f}".format(tolerance)
            tracking_info = "Tracker: {}".format(['off', 'on'][tracking])
            info = ", ".join([fps_info, tolerance_info, tracking_info])
            cv2.putText(frame,
                        text=info,
                        org=(10, 20),
                        fontFace=0,
                        fontScale=1e-3 * height,
                        color=(0, 0, 255),
                        thickness=thick)
            if tracking:
                if (len(TRACKERS) == 0) and (len(face_locations) > 0):
                    recognize_track_face(frame, tolerance)
                    print('recalculate')
                else:
                    for track in TRACKERS:
                        track.update(frame)
            else:
                recognize_face(frame, tolerance)
            cv2.imshow('track face', frame)

        keyboard = cv2.waitKey(1)
        # esc to quit
        if keyboard == 27:
            break
        # press m to flip frame
        if chr(keyboard & 255) == 'm':
            mirror = not mirror
        # switch tracker
        if chr(keyboard & 255) == 't':
            tracking = not tracking

    cv2.destroyAllWindows()


def main():
    show_face()


if __name__ == '__main__':
    main()
