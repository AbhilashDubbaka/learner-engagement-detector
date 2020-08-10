import cv2
import os
import sys

# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

def video_record(filename):
    frames_per_second = 30.0
    res = '720p'

    # Checks and deletes the output file
    # You cant have a existing file or it will through an error
    if os.path.isfile(filename):
        os.remove(filename)

    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)#VIDEO_PREDICTOR.camera_source
    out = cv2.VideoWriter(filename, get_video_type(filename), frames_per_second, get_dims(cap, res))

    while True:
        ret, frame = cap.read()

        if ret == True:
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    filename = sys.argv[1]
    video_record(filename)
