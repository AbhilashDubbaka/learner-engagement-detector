import numpy as np
import random as rn
from parameters import DATASET, NETWORK, OTHER
np.random.seed(OTHER.random_state)
rn.seed(OTHER.random_state)
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2
import dlib

%matplotlib inline

def face_detection(image_path):
    net = cv2.dnn.readNetFromCaffe(DATASET.config_file, DATASET.model_file)
    secondary_face_detector = dlib.get_frontal_face_detector()
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    only_face_image = image
    face_detected = False

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence <= DATASET.min_confidence:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        only_face_image = image[startY-2:endY+2, startX-2:endX+2]
        dlib_face_image = image[startY-5:endY+5, startX-5:endX+5].copy()
        dets = secondary_face_detector(dlib_face_image, 1)
        if(len(dets) == 0):
            continue
        face_detected = True
        break

    return only_face_image, face_detected

from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam
from vis.utils import utils
from keras import activations

model = load_model('models/AU1/AU1.h5')
penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_9")
#model = load_model('models/AU2/AU2a.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_26")
#model = load_model('models/AU2/AU2b.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_42")
#model = load_model('models/AU4/AU4.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_6")
#model = load_model('models/AU5/AU5.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_189")
#model = load_model('models/AU6/AU6.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_6")
#model = load_model('models/AU9/AU9.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_18")
#model = load_model('models/AU12/AU12.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_15")
#model = load_model('models/AU15/AU15.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_30")
#model = load_model('models/AU25/AU25.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_21")
#model = load_model('models/AU26/AU26.h5')
#penultimate_layer_idx = utils.find_layer_idx(model, "conv2d_6")

# Swap softmax with linear
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)

#Images
#image_path = "data/CKPlusDataset/cohn-kanade-images/S124/003/S124_003_00000011.png" #AU1
#image_path = "data/CKPlusDataset/cohn-kanade-images/S113/001/S113_001_00000012.png" #AU2
#image_path = "data/CKPlusDataset/cohn-kanade-images/S055/004/S055_004_00000028.png" #AU4
#image_path = "data/CKPlusDataset/cohn-kanade-images/S074/001/S074_001_00000020.png" #AU5
#image_path = "data/CKPlusDataset/cohn-kanade-images/S132/006/S132_006_00000023.png" #AU6
#image_path = "data/CKPlusDataset/cohn-kanade-images/S132/005/S132_005_00000016.png" #AU9
#image_path = "data/CKPlusDataset/cohn-kanade-images/S132/006/S132_006_00000023.png" #AU12
#image_path = "data/CKPlusDataset/cohn-kanade-images/S124/002/S124_002_00000017.png" #AU15
#image_path = "data/CKPlusDataset/cohn-kanade-images/S055/005/S055_005_00000045.png" #AU25
#image_path = "data/CKPlusDataset/cohn-kanade-images/S111/005/S111_005_00000011.png" #AU26

data, face_detected = face_detection(image_path)
data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
data = data[0:DATASET.upper_face_vertical_size, ].copy()
data = cv2.resize(data, (DATASET.face_horizontal_size, DATASET.upper_face_vertical_size), interpolation = cv2.INTER_AREA)
data = np.array(data).reshape(DATASET.upper_face_vertical_size, DATASET.face_horizontal_size, 1)

#data = cv2.resize(data, (DATASET.face_horizontal_size, DATASET.face_vertical_size))
#data = data[115:(115+DATASET.lower_face_vertical_size), ].copy()
#data = cv2.resize(data, (DATASET.face_horizontal_size, DATASET.lower_face_vertical_size), interpolation = cv2.INTER_AREA)
#data = np.array(data).reshape(DATASET.lower_face_vertical_size, DATASET.face_horizontal_size, 1)

#data = cv2.resize(data, (DATASET.face_horizontal_size, DATASET.face_vertical_size), interpolation = cv2.INTER_AREA)
#data = np.array(data).reshape(DATASET.face_vertical_size, DATASET.face_horizontal_size, 1)

data = data/255

# input tensor for model.predict
inp = data.reshape(1, DATASET.upper_face_vertical_size, DATASET.face_horizontal_size, 1)
img = data.reshape(DATASET.upper_face_vertical_size, DATASET.face_horizontal_size)

#inp = data.reshape(1, DATASET.lower_face_vertical_size, DATASET.face_horizontal_size, 1)
#img = data.reshape(DATASET.lower_face_vertical_size, DATASET.face_horizontal_size)

#inp = data.reshape(1, DATASET.face_vertical_size, DATASET.face_horizontal_size, 1)
#img = data.reshape(DATASET.face_vertical_size, DATASET.face_horizontal_size)
img = img*255
_ = plt.imshow(img, cmap="gray")

grads = visualize_cam(model, -1, filter_indices=0, 
                      seed_input=data, backprop_modifier='guided', penultimate_layer_idx = None)
plt.figure(figsize=(4.3, 4.3))
plt.imshow(img, cmap="gray")
plt.imshow(grads, cmap='jet', alpha = 0.3)
#plt.colorbar().solids.set(alpha=1)
plt.show()
