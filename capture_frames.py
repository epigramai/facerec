import os
import cv2
import matplotlib.pyplot as plt
from time import time, sleep

def capture_frames(framerate, duration, handler):
    cap = cv2.VideoCapture(0)
    sleep_time = 1 / framerate
    print('Sleep time: ' + str(sleep_time))
    num_frames = int(duration / sleep_time)
    print('Num frames: ' + str(num_frames))

    for i in range(num_frames):
        ret, frame = cap.read()
        handler(frame)
        sleep(sleep_time)

def _default_handler(folder):
    def handler(img):
        path = os.path.join(folder, '%d.jpg' % int(time()))

    return handler


if __name__ == '__main__':
    framerate = 0.5
    duration = 10
    capture_frames(framerate, duration, _default_handler('/Users/esten/'))