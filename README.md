# Learner Engagement Detector

A facial expression detector that classifies facial action units of a person in a video file and provides the valence, arousal and Learner Engagement levels throughout the video. This is the thesis that I completed as part of MSc Computing Science at Imperial College London. I published a modified version of this thesis in the IEEE EDUCON 2020 Conference in April 2020: [Detecting Learner Engagement in MOOCs using Automatic Facial Expression Recognition](https://ieeexplore.ieee.org/document/9125149).

### Abstract of Publication
Drop out rates in Massive Open Online Courses (MOOCs) are very high. Although there are many external factors, such as users not having enough time or not intending to complete the course, there are some aspects that instructors can control to optimally engage their students. To do this, they need to know how students engage throughout their video lecture. This paper explores the use of webcams to record students' facial expressions whilst they watched educational video material to analyse their Learner Engagement levels. Convolutional neural networks (CNNs) were trained to detect facial action units, which were mapped onto two psychological measurements, valence (emotional state) and arousal (attentiveness), using support vector regressions. These valence and arousal values were combined in a novel manner resulting in Learner Engagement levels.

Moreover, a new approach was used to combine CNNs with geometric feature-based techniques to improve the performance of the models. Two experiments were conducted and found that 9 out of 10 CNN models achieved 95% accuracy on average across the majority of the subjects, whilst the Learner Engagement detector was able to identify facial expressions that translated to Learner Engagement levels successfully. These results suggest that there is promise in this approach, in that feedback on students' Learner Engagement can be provided back to the instructor. Additional research should be undertaken to further prove these results and overcome some limitations that were faced.

### Installation
To start using this detector, install the python environment using either:
1) `environment_cpu.yml` or
2) `environment_gpu.yml`, if you have a GPU enabled on your machine

### Usage
A video file should be assigned through arguments and it should be a format that OpenCV supports (mp4, avi etc.):
```
python3 video_predictor.py <video_filename>
```

This returns a `.xlsx` file with all of the raw data from the analysis and 3 `.png` files, which show the valence, arousal and Learner Engagement chart over the duration of the video. This file is the Learner Engagement Detector as described in "System Design" from Chapter 3 of the report.

Note that the models, Excel files and datasets were not added to this repo since the external datasets were not allowed to be redistributed in any manner, which includes derived products such as the CNN models and the Excel files. The video recordings and images from the experiments were also not added due to data privacy. For use or demonstration of the models, please contact Abhilash Dubbaka (abhilash.dubbaka18@imperial.ac.uk).

### Graphical User Interfaces
4 different graphical user interfaces (GUIs) are provided in the `GUI` folder with the first two being used for the first experiment and the others being used for results analysis. This folder contains all the UI design files, which are imported into the GUIs, and the original `.ui` files that were created from the Qt Designer.

| File Usage | Detail |
| ------ | ------ |
| `python3 GUI/video_player_for_experiment.py <video_filename_to_save_recording_as>` | This enables the user to play a video by opening a video file in the GUI and once the play button is pressed, it starts automatically recording via the webcam. It saves to the video filename provided once the video finishes or the stop button is pressed.  |
| `python3 GUI/video_player_for_post_experiment.py`| This enables the user to play the original video watched by a person, alongside that person's webcam recording and are played in sync together. It does not allow playback if the files duration do not match. |
| `python3 GUI/video_player_for_results.py` | This enables the user to play the webcam recording once it is analysed by the video predictor and highlights the different action units activated throughout the video, as well as displaying the valence and arousal charts. The valence and arousal images as well as the Excel file have to be in the same directory as the video file. The original content can be played alongside the recording but this is optional. |
| `python3 GUI/video_player_for_engagement_results_individual.py` | This enables the user to play the webcam recording once it is analysed by the video predictor and displays the Learner Engagement chart, which has to be in the same directory as the video file. The original content can be played alongside the recording but this is optional. |

### Supporting files
| File | Detail |
| ------ | ------ |
| `data_organising.py` | Contains all the functions used to create the data pickle files for training the models from the datasets. Should be used in conjunction with `parameters.py` to choose the correct dataset for organising. |
| `grad_cam.py` | Through this file, created the gradient-weighted class activation maps (Grad-CAMs) output images to understand what the CNNs are looking for.|
| `head_pose_estimator.py` | A class that returns the head pose estimates based on 68-points facial landmarks provided by the user. |
| `model_training_tuning_AU.py` | Used for fine tuning the parameters for the CNN models through randomised searches and training them with the best parameters. Should be used in conjunction with `parameters.py` to choose the correct parameters for training. |
| `model_training_tuning_valence_arousal.py` | Used for fine tuning parameters for the SVR models through grid searches and training them with the best parameters. |
| `parameters.py` | Contains all the variables that are used across multiple files, in particular all the search parameters for the CNN models. |
| `photo_predictor_and_processor.py` | Predicts AUs and facial landmarks from an image. Also, contains functions to create the geometric thresholds used and predicting the action units for the valence and arousal dataset images. |
| `simple_video_recorder.py` | A simple webcam recorder with usage as `python3 simple_video_recorder <video_filename_to_save_recording_as>`. Compatible outputs are `.mp4` and `.avi`.|
| `utils.py` | Contains two useful functions to take pictures from a webcam and to extract frames from video. |