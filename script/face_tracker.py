"""
Recognize and track faces
"""
import time
import pickle
import cv2
import face_recognition
from image_labeler import label_object, label_info
from video_tracker import DlibCorrelationTracker, OpencvTracker

# pylint: disable=maybe-no-member
with open("face_data_dlib.pickle", "rb") as f_r:
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


def recognize_track_face(frame, tolerance, tracking, shrink=0.25):
    """
    Recognize and track faces
    """
    face_locations, face_encodings = detect_face(frame, shrink)
    tracks = []
    for location, face_encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = [int(i / shrink) for i in location]
        distances = face_recognition.face_distance(ENCODINGS, face_encoding)
        if min(distances) < tolerance:
            name = NAMES[distances.argmin()]
        else:
            name = "Unknown"
        if not tracking:
            label_object(frame, left, top, right, bottom, name)
            return None
        if tracking == "dlib":
            track = DlibCorrelationTracker(name, 5)
        else:
            track = OpencvTracker(name, tracking)
        track.init((left, top, right - left, bottom - top), frame)
        tracks.append(track)
    return tracks


def detect_face(frame, shrink):
    """
    Detect and encode face
    """
    small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    return face_locations, face_encodings


def clear_missed_face(frame, miss, shrink):
    """
    clear missed face trackers
    """
    small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
    face_locations = face_recognition.face_locations(small_frame)
    if len(face_locations) != len(TRACKERS):
        miss += 1
    if (len(face_locations) != len(TRACKERS)) and (miss >= 15):
        TRACKERS.clear()
        miss = 0
    return miss, len(face_locations)


def show_face(shrink=0.25):
    """
    Detect and recognize faces with webcam
    """
    cam = cv2.VideoCapture(0)
    tolerance = 0.5
    mirror = True
    counter = 0
    miss = 0
    tracker_type = [False, "dlib"]
    tracker_type.extend(list(CV_TRACKER.keys()))
    switch = 0
    tracking = False
    now = time.time()
    while True:
        ret_val, frame = cam.read()
        if not ret_val:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        counter += 1
        counter = counter % 10000

        if counter % 5 == 1:
            miss, face_num = clear_missed_face(frame, miss, shrink)
            fps = 5 / (time.time() - now)
            now = time.time()
            fps_info = "fps: {}".format(str(int(fps)).zfill(2))

        if tracking:
            if (len(TRACKERS) == 0) and (face_num > 0):
                track_objs = recognize_track_face(frame, tolerance, tracking, shrink)
                TRACKERS.extend(track_objs)
            else:
                for track_obj in TRACKERS:
                    track_obj.update(frame)
        else:
            recognize_track_face(frame, tolerance, tracking, shrink)

        tolerance_info = "tolerance: {:.2f}".format(tolerance)
        tracking_info = "Tracker: {}".format(tracking)
        button_info = (
            "Press t to switch tracker type. Press m to flip image. Presss ESC to quit."
        )
        label_info(frame, button_info, fps_info, tolerance_info, tracking_info)

        cv2.namedWindow("Track", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Track", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Track", frame)

        keyboard = cv2.waitKey(1)
        if keyboard == 27:
            break
        if chr(keyboard & 255) == "m":
            mirror = not mirror
        if chr(keyboard & 255) == "t":
            TRACKERS.clear()
            switch += 1
            switch %= len(tracker_type)
            tracking = tracker_type[switch]
    cam.release()
    cv2.destroyAllWindows()


def main():
    """
    Recognize and track faces
    """
    show_face(0.25)


if __name__ == "__main__":
    print("Press t to switch tracker type. Press m to flip image. Presss ESC to quit.")
    main()
