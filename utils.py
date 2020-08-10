import cv2

#Function to take pictures from webcam
cam = cv2.VideoCapture(0)
cv2.namedWindow("Webcam Image")
img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("Webcam Image", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()
cv2.destroyAllWindows()

#Function to extract frames from video
vidcap = cv2.VideoCapture("video_recordings/Experiment_1/Ieva_1.avi")
success,image = vidcap.read()
count = 0
while success:
    count += 1
    cv2.imwrite("video_recordings/Experiment_1/Ieva/frame%d.jpg" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success, count)
vidcap.release()