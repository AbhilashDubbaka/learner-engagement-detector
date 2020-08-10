from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os
import pandas as pd
import math
import sys
from UI.main_window_for_results import Ui_MainWindow
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from parameters import VIDEO_PREDICTOR

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
        self.df = None
        self.mapping_video_index_to_AU = {0:self.AU1, 1:self.AU2, 2:self.AU4, 3:self.AU5, 4:self.AU6,
                                 5:self.AU9, 6:self.AU12, 7:self.AU15, 8:self.AU25, 9:self.AU26,
                                  10:self.EAR}
        self.mapping_video_index_to_AU_labels = {0:self.AU1_label, 1:self.AU2_label, 2:self.AU4_label, 
                                                 3:self.AU5_label, 4:self.AU6_label, 5:self.AU9_label, 
                                                 6:self.AU12_label, 7:self.AU15_label, 8:self.AU25_label, 
                                                 9:self.AU26_label, 10:self.drowsiness_label}

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
                    excel = filename + ".xlsx"
                    if os.path.isfile(excel):
                        self.df = pd.read_excel(excel, "processed_data")
                        pixmap_valence = QPixmap(filename + '_valence.png')
                        pixmap_valence = pixmap_valence.scaled(795, 121)
                        pixmap_arousal = QPixmap(filename + '_arousal.png')
                        pixmap_arousal = pixmap_arousal.scaled(795, 121)
                        self.valence_chart.setPixmap(pixmap_valence)
                        self.arousal_chart.setPixmap(pixmap_arousal)
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
        self.arousalSlider.setMaximum(duration)
        self.valenceSlider.setMaximum(duration)

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
       
        current_time_in_half_second = round(0.5 * math.floor(float(position/1000)/0.5),1)
        info_row = self.df.iloc[int(current_time_in_half_second*2)]
        self.update_AU_labels(info_row)

        # Disable the events to prevent updating triggering a setPosition event (can cause stuttering).
        self.timeSlider.blockSignals(True)
        self.timeSlider.setValue(position)
        self.timeSlider.blockSignals(False)
        
        self.arousalSlider.blockSignals(True)
        self.arousalSlider.setValue(position)
        self.arousalSlider.blockSignals(False)
        
        self.valenceSlider.blockSignals(True)
        self.valenceSlider.setValue(position)
        self.valenceSlider.blockSignals(False)
        
    def update_AU_labels(self, info_row):
        for i in range(1, 11):
            color = "black"
            if info_row[i] >= 0.6:
                color = "red"
            self.mapping_video_index_to_AU[i-1].setText("<html><head/><body><p><span style=\" font-size:9pt; color:%s\">%s</span></p></body></html>" %(color, '%.2f' % info_row[i]))
            self.mapping_video_index_to_AU_labels[i-1].setText("<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; color: %s\">%s</span></p></body></html>" %(color, VIDEO_PREDICTOR.mapping_video_index_to_AU_label_text[i-1]))
        
        color = "black"    
        if info_row[12] == "Yes":
            color = "red"
        self.mapping_video_index_to_AU[10].setText("<html><head/><body><p><span style=\" font-size:9pt; color:%s\">%s</span></p></body></html>" %(color, '%.2f' % info_row[11]))
        self.mapping_video_index_to_AU_labels[10].setText("<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; color: %s\">%s</span></p></body></html>" %(color, VIDEO_PREDICTOR.mapping_video_index_to_AU_label_text[10]))
    
    def erroralert(self, *args):
        print(args)

if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("Video Player")
    window = MainWindow()
    window.show()
    app.exec_()
