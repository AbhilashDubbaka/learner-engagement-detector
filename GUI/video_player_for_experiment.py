from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import cv2
import os
import sys
from main_window_for_experiment import Ui_MainWindow

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def hhmmss(ms):
    # s = 1000
    # m = 60000
    # h = 360000
    h, r = divmod(ms, 360000)
    m, r = divmod(r, 60000)
    s, _ = divmod(r, 1000)
    return ("%d:%02d:%02d" % (h,m,s)) if h else ("%d:%02d" % (m,s))

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.player = QMediaPlayer()
        self.player.error.connect(self.erroralert)
        self.player.setVideoOutput(self.videoWidget)
        self.capture = Capture()

        # Connect control buttons/slides for media player.
        self.playButton.clicked.connect(self.start)
        self.pauseButton.clicked.connect(self.pause)
        self.stopButton.clicked.connect(self.exitCall)
        self.volumeSlider.valueChanged.connect(self.player.setVolume)

        self.player.durationChanged.connect(self.update_duration)
        self.player.positionChanged.connect(self.update_position)
        # self.timeSlider.valueChanged.connect(self.player.setPosition) #This to disable being able to move the slider

        self.open_file_action.triggered.connect(self.open_file)

    def start(self):
        self.player.play()
        if self.capture.capturing == False:
            self.capture.startCapture()
        if self.capture.paused == True:
            self.capture.paused = False

    def exitCall(self):
        self.player.stop()
        self.capture.quitCapture()
        # RIGHT TO FILE EXCEL OF FRAMES PAUSED ETC
        QCoreApplication.quit()

    def pause(self):
        self.player.pause()
        if self.capture.paused == False:
            print(hhmmss(self.player.position()))
            self.capture.paused = True

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open file", QDir.homePath())

        if path != '':
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
            self.playButton.setEnabled(True)
            self.pauseButton.setEnabled(True)
            self.stopButton.setEnabled(True) # Commenting this disables the stop button for experiments

    def update_duration(self, duration):
        print("!", duration)
        print("?", self.player.duration())

        self.timeSlider.setMaximum(duration)

        if duration >= 0:
            self.totalTimeLabel.setText(hhmmss(duration))

    def update_position(self, position):
        if position >= 0:
            print(position)
            self.currentTimeLabel.setText(hhmmss(position))

        # Disable the events to prevent updating triggering a setPosition event (can cause stuttering).
        self.timeSlider.blockSignals(True)
        self.timeSlider.setValue(position)
        self.timeSlider.blockSignals(False)

        if position == self.player.duration() and self.capture.capturing == True:
#            time.sleep(2)
            self.capture.capturing = False

    def erroralert(self, *args):
        print(args)


class Capture():
   def __init__(self):
       self.capturing = False
       self.paused = False
       self.cap = cv2.VideoCapture(0)
       self.filename = sys.argv[1]
       # Checks and deletes the output file
       if os.path.isfile(self.filename):
           os.remove(self.filename)
       self.frames_per_second = 30.0
       self.res = '720p'
       self.out = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*'XVID'), self.frames_per_second, self.get_dims())

   def startCapture(self):
       print ("pressed start")
       self.capturing = True
       cap = self.cap
       frame_count = 0
       while(self.capturing):
           ret, frame = cap.read()
           if ret == True:
               frame_count += 1
               if self.paused == True:
                   print(frame_count)
               if self.paused == False:
                   self.out.write(frame)
               # cv2.imshow('frame',frame)
               cv2.waitKey(0)
       self.quitCapture()


   def quitCapture(self):
       print ("pressed Quit")
       self.capturing = False
       self.cap.release()
       self.out.release()
       cv2.destroyAllWindows()

   # Set resolution for the video capture
   def change_res(self, width, height):
       self.cap.set(3, width)
       self.cap.set(4, height)

   # grab resolution dimensions and set video capture to it.
   def get_dims(self):
       width, height = STD_DIMENSIONS["480p"]
       if self.res in STD_DIMENSIONS:
           width,height = STD_DIMENSIONS[self.res]
       ## change the current caputre device
       ## to the resulting resolution
       self.change_res(width, height)
       return width, height

if __name__ == '__main__':
    app = QApplication(sys.argv) #sys.argv
    app.setApplicationName("Video Player")
    window = MainWindow()
    window.show()
    app.exec_()
