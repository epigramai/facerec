import cv2
import os
import sys
from time import time, sleep
import matplotlib.pyplot as plt

from facedetector import FaceDetector

DEFAULT_FRAMERATE = 2
DEFAULT_DURATION = 5
DETECT_FACES = True

def capture_frames(framerate, duration, detect_faces):
    if detect_faces:
        detector = FaceDetector()
        return
    cap = cv2.VideoCapture(0)
    sleep_time = 1 / framerate
    num_frames = int(duration / sleep_time)

    for i in range(num_frames):
        ret, frame = cap.read()

        if detect_faces:
            detector.detect(frame)

        cv2.imshow('Stream', frame)
        cv2.waitKey(1)
        sleep(sleep_time)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        framerate = float(sys.argv[1])
    else:
        framerate = DEFAULT_FRAMERATE

    if len(sys.argv) > 2:
        duration = float(sys.argv[2])
    else:
        duration = DEFAULT_DURATION

    capture_frames(framerate, duration, DETECT_FACES)