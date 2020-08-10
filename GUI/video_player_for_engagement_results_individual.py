from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os
from UI.main_window_for_engagement_results_individual import Ui_MainWindow

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
        self.player.setVideoOutput(self.videoWidget_2)
        self.player.setNotifyInterval(1)
        
        self.player2 = QMediaPlayer()
        self.player2.error.connect(self.erroralert)
        self.player2.setVideoOutput(self.videoWidget)
        self.player2.setNotifyInterval(1)
        
        self.selected_viewer_reaction_video = False
        self.selected_original_content_video = False
        self.selected_viewer_reaction_video_duration = None
        self.selected_original_content_video_duration = None
       
        # Connect control buttons/slides for media player.
        self.playButton.clicked.connect(self.start)
        self.pauseButton.clicked.connect(self.pause)
        self.stopButton.clicked.connect(self.exitCall)
        self.volumeSlider.valueChanged.connect(self.player2.setVolume)

        self.player.durationChanged.connect(self.update_duration)
        self.player2.durationChanged.connect(self.update_duration2)
        self.player.positionChanged.connect(self.update_position)
        self.timeSlider.valueChanged.connect(self.player.setPosition)
        self.timeSlider.valueChanged.connect(self.player2.setPosition)

        self.open_file_action.triggered.connect(lambda: self.open_file("original"))
        self.open_viewer_reaction_file_action.triggered.connect(lambda: self.open_file("viewer"))
        
    def start(self):
        self.player.play()
        self.player2.play()

    def exitCall(self):
        self.player.stop()
        self.player2.stop()

    def pause(self):
        self.player.pause()
        self.player2.pause()

    def open_file(self, video_type):
        valid_file = False
        while valid_file == False:
            path, _ = QFileDialog.getOpenFileName(self, "Open file", QDir.homePath())
            if path != '': 
                if video_type == "viewer":
                    filename, ext = os.path.splitext(path)
                    engagement = filename + '_engagement.png'
                    if os.path.isfile(engagement):
                        pixmap_engagement = QPixmap(engagement)
                        pixmap_engagement = pixmap_engagement.scaled(770, 281)
                        self.engagement_chart.setPixmap(pixmap_engagement)
                        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
                        self.selected_viewer_reaction_video = True
                        self.playButton.setEnabled(True)
                        self.pauseButton.setEnabled(True)
                        self.stopButton.setEnabled(True)
                        valid_file = True
                    else:
                        error_box = QMessageBox.critical(self, 'Video Selection Error',
                                            ("You have chosen an unanalysed video. Please select another video."),
                                            QMessageBox.Ok | QMessageBox.Cancel)
                        if error_box == QMessageBox.Ok:
                            pass
                        else:
                            valid_file = True
                
                else:
                    self.player2.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
                    self.selected_original_content_video = True
                    valid_file = True
            else:
                valid_file = True
    
    def update_duration(self, duration):
        print("!", duration)
        print("?", self.player.duration())

        self.timeSlider.setMaximum(duration)
        self.engagementSlider.setMaximum(duration)

        if duration >= 0:
            self.totalTimeLabel.setText(hhmmss(duration))
        
        self.selected_viewer_reaction_video_duration = duration
        self.check_duration_of_videos()
    
    def update_duration2(self, duration):       
        self.selected_original_content_video_duration = duration
        self.check_duration_of_videos()
    
    def check_duration_of_videos(self):
        if self.selected_viewer_reaction_video == True and self.selected_original_content_video == True:
            if (abs(self.selected_viewer_reaction_video_duration - self.selected_original_content_video_duration) >= 1000
                    and self.selected_viewer_reaction_video_duration != 0 
                    and self.selected_original_content_video_duration != 0):
                error_box = QMessageBox.critical(self, 'Video Selection Error',
                                            ("The original content video does not correspond to the viewer reaction video. \n" 
                                             "1) Please press Ok if would you like to choose a different video for the original content. \n"
                                             "2) Please press Retry if would you like to choose a different video for the viewer reaction. \n"
                                             "3) Alternatively, press Ignore to continue with the viewer reaction video," 
                                             " but note a video will not be played for the original content."),
                                            QMessageBox.Ok | QMessageBox.Retry | QMessageBox.Ignore)
                if error_box == QMessageBox.Ok:
                    self.player2.setMedia(QMediaContent())
                    self.open_file("original")
                elif error_box == QMessageBox.Retry:
                    self.player.setMedia(QMediaContent())
                    self.playButton.setEnabled(False)
                    self.pauseButton.setEnabled(False)
                    self.stopButton.setEnabled(False)
                    self.open_file("viewer")
                    
                else:
                    self.player2.setMedia(QMediaContent())
    
    def update_position(self, position):
        if position >= 0:
            self.currentTimeLabel.setText(hhmmss(position))

        # Disable the events to prevent updating triggering a setPosition event (can cause stuttering).
        self.timeSlider.blockSignals(True)
        self.timeSlider.setValue(position)
        self.timeSlider.blockSignals(False)
        
        self.engagementSlider.blockSignals(True)
        self.engagementSlider.setValue(position)
        self.engagementSlider.blockSignals(False)
         
    def erroralert(self, *args):
        print(args)

if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("Video Player")
    window = MainWindow()
    window.show()
    app.exec_()
