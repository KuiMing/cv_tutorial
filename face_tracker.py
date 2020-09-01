import time
import pickle
import cv2
import face_recognition

# pylint: disable=maybe-no-member
with open("face_data.pickle", "rb") as f_r:
    NAMES, ENCODINGS = pickle.load(f_r)
f_r.close()


def recognize_face(frame, tolerance):
    shrink = 0.25
    small_frame = cv2.resize(frame, (0, 0), fx=shrink, fy=shrink)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame,
                                                     face_locations)
    for location, face_encoding in zip(face_locations, face_encodings):

        top, right, bottom, left = [i * 4 for i in location]

        distances = face_recognition.face_distance(ENCODINGS, face_encoding)
        if min(distances) < tolerance:
            name = NAMES[distances.argmin()]
        else:
            name = "Unknown"
        recognition = "{name} {dist:.2f}".format(name=name,
                                                 dist=min(distances))
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
                    text=recognition,
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

            tolerance_info = "tolerance: {:.2f}".format(tolerance)
            info = ", ".join([fps_info, tolerance_info])
            cv2.putText(frame,
                        text=info,
                        org=(10, 20),
                        fontFace=0,
                        fontScale=1e-3 * height,
                        color=(0, 0, 255),
                        thickness=thick)
            recognize_face(frame, tolerance)
            cv2.imshow('web stream', frame)

        keyboard = cv2.waitKey(1)
        # esc to quit
        if keyboard == 27:
            break
        # press m to flip frame
        if chr(keyboard & 255) == 'm':
            mirror = not mirror

    cv2.destroyAllWindows()


def main():
    show_face()


if __name__ == '__main__':
    main()
