import numpy as np
import random as rn
from parameters import DATASET, TRAINING, OTHER
np.random.seed(OTHER.random_state)
rn.seed(OTHER.random_state)
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import talos as ta
import cv2
from fpdf import FPDF
import dlib

#%matplotlib inline

def import_pickle_data(dataset):

    with open("processed_data/" + dataset[0] + ".pickle", "rb") as pickle_in:
        AU_data = pickle.load(pickle_in)

    for k in range(1, len(dataset)):
        with open("processed_data/" + dataset[k] + ".pickle", "rb") as temp_pickle:
            temp = pickle.load(temp_pickle)
        for i in range(len(AU_data)):
            for j in range(len(temp[i])):
                AU_data[i].append(temp[i][j])

    with open("processed_data/CK+.pickle", "rb") as temp_pickle:
        temp = pickle.load(temp_pickle)
    for i in range(len(AU_data)):
        index = DATASET.mapping_CK_to_DISFA[i]
        for j in range(len(temp[index])):
            AU_data[i].append(temp[index][j])

    return AU_data

def extract_train_test_for_AU(AU_data, index):

    rn.shuffle(AU_data[index])
    x = []
    y = []
    for features,label in AU_data[index]:
        x.append(features)
        y.append(label)

    if index in DATASET.upper_facial_AUs:
        for i in range(len(x)):
            x[i] = x[i][0:(DATASET.upper_face_vertical_size), ]
            x[i] = cv2.resize(x[i], (DATASET.face_horizontal_size, DATASET.upper_face_vertical_size))
        x = np.array(x).reshape(-1, DATASET.upper_face_vertical_size, DATASET.face_horizontal_size, 1)
    elif index in DATASET.lower_facial_AUs:
        for i in range(len(x)):
            x[i] = x[i][115:(115+DATASET.lower_face_vertical_size), ]
            x[i] = cv2.resize(x[i], (DATASET.face_horizontal_size, DATASET.lower_face_vertical_size))
        x = np.array(x).reshape(-1, DATASET.lower_face_vertical_size, DATASET.face_horizontal_size, 1)
    elif index in DATASET.whole_facial_AUs:
        x = np.array(x).reshape(-1, DATASET.face_vertical_size, DATASET.face_horizontal_size, 1)

    y = np.array(y)

    # Normalize image vectors
    x = x/255

    # Convert labels to one hot matrices
    y = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = TRAINING.test_size, random_state = OTHER.random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = TRAINING.val_size, random_state = OTHER.random_state)
    return x_train, x_val, x_test, y_train, y_val, y_test

def create_model(x_train, y_train, x_val, y_val, params):

    index = DATASET.mapping_AU_to_Index[TRAINING.action_unit]
    # initialize the input shape and channel dimension
    if index in DATASET.upper_facial_AUs:
        input_shape = (DATASET.upper_face_vertical_size, DATASET.face_horizontal_size, 1)
    elif index in DATASET.lower_facial_AUs:
        input_shape = (DATASET.lower_face_vertical_size, DATASET.face_horizontal_size, 1)
    elif index in DATASET.whole_facial_AUs:
        input_shape = (DATASET.face_vertical_size, DATASET.face_horizontal_size, 1)
    chanDim = -1

    f_m_1 = params['feature_maps_1']
    f_m_2 = params['feature_maps_1'] * 2
    f_m_3 = params['feature_maps_1'] * 4
    f_c_1 = 64
    f_c_2 = 64

    if params['fc_neurons_1'] == 'fc1_64':
        f_c_1 = 64
    if params['fc_neurons_1'] == 'fc1_128':
        f_c_1 = 128
    if params['fc_neurons_1'] == 'fc1_256':
        f_c_1 = 256
    if params['fc_neurons_1'] == 'fc1_512':
        f_c_1 = 512

    if params['fc_neurons_2'] == 'fc2_64':
        f_c_2 = 64
    if params['fc_neurons_2'] == 'fc2_128':
        f_c_2 = 128
    if params['fc_neurons_2'] == 'fc2_256':
        f_c_2 = 256
    if params['fc_neurons_2'] == 'fc2_512':
        f_c_2 = 512

    model=Sequential()
    model.add(Conv2D(f_m_1, (3, 3), padding="same", kernel_initializer='he_normal', input_shape=input_shape))
    if params['batch_before_after'] == 'before' and params['batch_norm_1'] == 'bn1_1':
        model.add(BatchNormalization(axis=chanDim, momentum = 0.8))
    model.add(Activation("relu"))
    if params['batch_before_after'] == 'after' and params['batch_norm_1'] == 'bn1_1':
        model.add(BatchNormalization(axis=chanDim, momentum = 0.8))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(f_m_2, (3, 3), padding="same", kernel_initializer='he_normal'))
    if params['batch_before_after'] == 'before' and params['batch_norm_2'] == 'bn2_1':
        model.add(BatchNormalization(axis=chanDim, momentum = 0.8))
    model.add(Activation("relu"))
    if params['batch_before_after'] == 'after' and params['batch_norm_2'] == 'bn2_1':
        model.add(BatchNormalization(axis=chanDim, momentum = 0.8))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    if params['conv_layers'] == 3:
        model.add(Conv2D(f_m_3, (3, 3), padding="same", kernel_initializer='he_normal'))
        if params['batch_before_after'] == 'before' and params['batch_norm_3'] == 'bn3_1':
            model.add(BatchNormalization(axis=chanDim, momentum = 0.8))
        model.add(Activation("relu"))
        if params['batch_before_after'] == 'after' and params['batch_norm_3'] == 'bn3_1':
            model.add(BatchNormalization(axis=chanDim, momentum = 0.8))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(f_c_1, kernel_initializer='he_normal'))
    if params['batch_before_after'] == 'before' and params['batch_norm_4'] == 'bn4_1':
        model.add(BatchNormalization(momentum = 0.8))
    model.add(Activation("relu"))
    if params['batch_before_after'] == 'after' and params['batch_norm_4'] == 'bn4_1':
        model.add(BatchNormalization(momentum = 0.8))
    if params['fully_connected_layers'] == 2:
        model.add(Dense(f_c_2, kernel_initializer='he_normal'))
        if params['batch_before_after'] == 'before' and params['batch_norm_5'] == 'bn5_1':
            model.add(BatchNormalization(momentum = 0.8))
        model.add(Activation("relu"))
        if params['batch_before_after'] == 'after' and params['batch_norm_5'] == 'bn5_1':
            model.add(BatchNormalization(momentum = 0.8))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    # compile the model
    print("[INFO] compiling model...")
    if params['optimizer'] == 'Adam':
        opt = Adam(lr = params['learning_rate'])
    if params['optimizer'] == 'SGD':
        opt = SGD(lr = params['learning_rate'] * 10)
    if params['optimizer'] == 'SGD+N':
        opt = SGD(lr = params['learning_rate'] * 10, momentum = params['momentum'], nesterov=True)
    if params['optimizer'] == 'RMSprop':
        opt = RMSprop(lr = params['learning_rate'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])

    #get the length of the train and validation data
    ntrain = len(x_train)
    nval = len(x_val)

    train_datagen = ImageDataGenerator(rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                                       horizontal_flip=True)  # randomly flip images
    train_generator = train_datagen.flow(x_train, y_train, batch_size = params['batch_size'])

    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    if params['reduceLR'] == 1:
        history = model.fit_generator(train_generator,
                                      steps_per_epoch = ntrain // params['batch_size'],
                                      epochs = params['epochs'],
                                      callbacks = [rlrop],
                                      validation_data = (x_val, y_val),
                                      validation_steps = nval // params['batch_size'])
    else:
        history = model.fit_generator(train_generator,
                                      steps_per_epoch = ntrain // params['batch_size'],
                                      epochs = params['epochs'],
                                      validation_data = (x_val, y_val),
                                      validation_steps = nval // params['batch_size'])

    return history, model

def create_model_from_best_params(x_train, y_train, x_val, y_val, params, index, model_filename, i):
    # initialize the input shape and channel dimension
    if index in DATASET.upper_facial_AUs:
        input_shape = (DATASET.upper_face_vertical_size, DATASET.face_horizontal_size, 1)
    elif index in DATASET.lower_facial_AUs:
        input_shape = (DATASET.lower_face_vertical_size, DATASET.face_horizontal_size, 1)
    elif index in DATASET.whole_facial_AUs:
        input_shape = (DATASET.face_vertical_size, DATASET.face_horizontal_size, 1)
    chanDim = -1

    f_m_1 = params['feature_maps_1'][i]
    f_m_2 = f_m_1 * 2
    f_m_3 = f_m_1 * 4
    f_c_1 = params['fc_neurons_1'][i]
    f_c_2 = params['fc_neurons_2'][i]
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0.2
    d5 = 0.2

    model = Sequential()
    model.add(Conv2D(f_m_1, (3, 3), padding="same", kernel_initializer='he_normal', input_shape=input_shape))
    model.add(BatchNormalization(axis=chanDim, momentum = 0.8))
    model.add(Activation("relu"))
    model.add(Dropout(d1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(f_m_2, (3, 3), padding="same", kernel_initializer='he_normal'))
    model.add(BatchNormalization(axis=chanDim, momentum = 0.8))
    model.add(Activation("relu"))
    model.add(Dropout(d2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(f_m_3, (3, 3), padding="same", kernel_initializer='he_normal'))
    model.add(BatchNormalization(axis=chanDim, momentum = 0.8))
    model.add(Activation("relu"))
    model.add(Dropout(d3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(f_c_1, kernel_initializer='he_normal'))
    model.add(BatchNormalization(momentum = 0.8))
    model.add(Activation("relu"))
    model.add(Dropout(d4))
    model.add(Dense(f_c_2, kernel_initializer='he_normal'))
    model.add(Activation("relu"))
    model.add(Dropout(d5))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    # compile the model
    print("[INFO] compiling model...")
    if params['optimizer'][i] == 'Adam':
        opt = Adam()
    if params['optimizer'][i] == 'RMSprop':
        opt = RMSprop()
    if params['optimizer'][i] == 'SGD+N':
        opt = SGD(momentum = params['momentum'][i], nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])

    # Get the length of the train and validation data
    ntrain = len(x_train)
    nval = len(x_val)

    train_datagen = ImageDataGenerator(rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                                       horizontal_flip=True)  # randomly flip images
    train_generator = train_datagen.flow(x_train, y_train, batch_size = 32)

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    history = model.fit_generator(train_generator,
                                  steps_per_epoch = ntrain // 32,
                                  epochs = 15, #reaches quite high with only 8 epochs so only increase by 7 more
                                  callbacks = [early_stopping_callback, rlrop],
                                  validation_data = (x_val, y_val),
                                  validation_steps = nval // 32)

    model.save(model_filename + ".h5")
    return history.history

def new_data_for_AU_detection():
    training_data_AUs = []
    for i in range(1, 3): 
        image, face_detected = face_detection("processed_data/" + TRAINING.action_unit +"/frame_{}.jpg".format(i))
        if face_detected == False:
            print("No face detected in ", i)
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        whole_face = cv2.resize(image_grayscale, (DATASET.face_horizontal_size, DATASET.face_vertical_size))  # resize to normalize data size
        if DATASET.mapping_AU_to_Index[TRAINING.action_unit] in DATASET.upper_facial_AUs:
            face = whole_face[0:DATASET.upper_face_vertical_size, ].copy()
        elif DATASET.mapping_AU_to_Index[TRAINING.action_unit] in DATASET.lower_facial_AUs:
            face = whole_face[115:(115+DATASET.lower_face_vertical_size), ].copy()
        else:
            face = whole_face
        
        training_data_AUs.append([face])

    return training_data_AUs

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
        only_face_image = image[startY+2:endY-2, startX+2:endX-2]
        dlib_face_image = image[startY-5:endY+5, startX-5:endX+5].copy()
        dets = secondary_face_detector(dlib_face_image, 1)
        if(len(dets) == 0):
            continue
        face_detected = True
        break

    return only_face_image, face_detected

def evaluate_model(model_filename, history, x_test, y_test):
    model = load_model(model_filename + ".h5")

    #Evaluation on testset
    preds = model.evaluate(x_test, y_test)

    #Gather the predictions to get a classification report
    y_pred = model.predict(x_test, verbose=2)
    report = classification_report(y_test, y_pred.round())

    #Transform test and predictions to obtain a confusion matrix
    y_test_non_category = [np.argmax(t) for t in y_test]
    y_predict_non_category = [np.argmax(t) for t in y_pred]
    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)

    #Test model on personal data
    data = new_data_for_AU_detection()
    if DATASET.mapping_AU_to_Index[TRAINING.action_unit] in DATASET.upper_facial_AUs:
        data = np.array(data).reshape(-1, DATASET.upper_face_vertical_size, DATASET.face_horizontal_size, 1)
    elif DATASET.mapping_AU_to_Index[TRAINING.action_unit] in DATASET.lower_facial_AUs:
        data = np.array(data).reshape(-1, DATASET.lower_face_vertical_size, DATASET.face_horizontal_size, 1)
    else:
        data = np.array(data).reshape(-1, DATASET.face_vertical_size, DATASET.face_horizontal_size, 1)
    data = data/255
    y_pred2 = model.predict(data, verbose=2)

    #Save evaluations
    saved_data = []
    saved_data.append(history)
    saved_data.append(preds)
    saved_data.append(report)
    saved_data.append(conf_mat)
    saved_data.append(y_pred2)

    with open(model_filename + '_evaluation.pickle', 'wb') as f:
        pickle.dump(saved_data, f)

def view_evaluation_of_model(model_filename, pdf):
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.write(5, model_filename)
    pdf.ln()

    with open(model_filename + '_evaluation.pickle', 'rb') as f:
        history, preds, report, conf_mat, y_pred2 = pickle.load(f)

    #Plot the train and val curve
    epochs = range(1, len(history['acc']) + 1)
    #Train and validation accuracy
    plt.figure()
    plt.plot(epochs, history['acc'], 'b', label='Training accurarcy')
    plt.plot(epochs, history['val_acc'], 'r', label='Validation accurarcy')
    plt.title('Training and Validation accuracy of ' + model_filename)
    plt.legend()
    plt.savefig(model_filename + '_plot.png')
    pdf.image(model_filename + '_plot.png', w = 108 , h = 72)
    plt.close()
    plt.figure()
    #Train and validation loss
    plt.plot(epochs, history['loss'], 'b', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation loss')
    plt.title('Training and Validation loss of ' + model_filename)
    plt.legend()
    plt.savefig(model_filename + '_plot2.png')
    pdf.image(model_filename + '_plot2.png', w = 108 , h = 72)
    plt.close()

    pdf.write(5, "Test Loss = " + str(preds[0]))
    pdf.ln()
    pdf.write(5, "Test Accuracy = " + str(preds[1]))
    pdf.ln()
    pdf.write(5, report)
    pdf.ln()
    for i in range(0, len(conf_mat)):
        pdf.cell(40, 5, '{}'.format(conf_mat[i][0]), 1, 0, 'C')
        pdf.cell(40, 5, '{}'.format(conf_mat[i][1]), 1, 1, 'C')
    pdf.ln()
    for i in range(0, len(y_pred2)):
        pdf.cell(60, 5, '{}'.format(y_pred2[i][0]), 1, 0, 'C')
        pdf.cell(60, 5, '{}'.format(y_pred2[i][1]), 1, 1, 'C')

def print_evaluations(models):
    pdf = FPDF()
    for i in range(len(models)):
        model_filename = 'models/' + TRAINING.action_unit +'/' + models[i]
        view_evaluation_of_model(model_filename, pdf)
        print(i)
    pdf.output('models/' + TRAINING.action_unit + '/model_evaluations.pdf', 'F')

#################################################################################
# Main functions
dataset = TRAINING.dataset


############################## Experiments #####################################
AU_data = import_pickle_data(dataset)
x_train, x_val, x_test, y_train, y_val, y_test = extract_train_test_for_AU(AU_data, DATASET.mapping_AU_to_Index[TRAINING.action_unit])

#Experiment 1
t = ta.Scan(x = x_train,
           y = y_train,
           x_val = x_val,
           y_val = y_val,
           model = create_model,
           params = TRAINING.first_params,
           experiment_no = TRAINING.action_unit + '_Exp_1',
           grid_downsample = 0.0204,
           )

#Experiment 1 Results Analysis
r = ta.Reporting("models/" + TRAINING.action_unit + "/" + TRAINING.action_unit + "_Exp_1.csv")
r.plot_corr('val_acc')

#Experiment 2
t = ta.Scan(x = x_train,
          y = y_train,
          x_val = x_val,
          y_val = y_val,
          model = create_model,
          params = TRAINING.second_params,
          experiment_no = TRAINING.action_unit + '_Exp_2',
          grid_downsample = 0.0408,
          )

#Experiment 2 Results Analysis
r = ta.Reporting("models/" + TRAINING.action_unit + "/" + TRAINING.action_unit + "_Exp_2.csv")
r.plot_corr('val_acc')
r.plot_bars('fully_connected_layers', 'val_acc', 'batch_before_after', 'conv_layers')

#Experiment 3
t = ta.Scan(x = x_train,
        y = y_train,
        x_val = x_val,
        y_val = y_val,
        model = create_model,
        params = TRAINING.third_params,
        experiment_no = TRAINING.action_unit + '_Exp_3',
        grid_downsample = 0.3265,
        )

#Experiment 3 Results Analysis
r = ta.Reporting("models/" + TRAINING.action_unit + "/" + TRAINING.action_unit + "_Exp_3.csv")
r.plot_kde('val_acc', 'conv_layers')
r.plot_kde('val_acc', 'batch_norm_2')
r.plot_kde('val_acc', 'batch_norm_3')
r.plot_kde('val_acc', 'batch_norm_5')

#Experiment 4
t = ta.Scan(x = x_train,
          y = y_train,
          x_val = x_val,
          y_val = y_val,
          model = create_model,
          params = TRAINING.fourth_params,
          experiment_no = TRAINING.action_unit + '_Exp_4',
          grid_downsample = 0.0163,
          )

#Experiment 4 Results Analysis
r = ta.Reporting("models/" + TRAINING.action_unit + "/" + TRAINING.action_unit + "_Exp_4.csv")
r.plot_bars('learning_rate', 'val_acc', 'optimizer', 'batch_size')

#SGD Momentum Experiment
t = ta.Scan(x = x_train,
          y = y_train,
          x_val = x_val,
          y_val = y_val,
          model = create_model,
          params = TRAINING.mom_params,
          experiment_no = TRAINING.action_unit + '_Exp_Mom',
          grid_downsample = 1,
          )

#Adam and RMSprop Experiment
t = ta.Scan(x = x_train,
          y = y_train,
          x_val = x_val,
          y_val = y_val,
          model = create_model,
          params = TRAINING.opt_params,
          experiment_no = TRAINING.action_unit + '_Exp_Opt',
          grid_downsample = 1,
          )

#Final Results Analysis after combining csv of SGD and Adam and RMSprop experiments
r = ta.Reporting("models/" + TRAINING.action_unit + "/" + TRAINING.action_unit + "_Exp_Opt.csv") #manually combine csv of Opt and Mom files
r.plot_corr('val_acc')
r.correlate('val_acc')
r.best_params()

############################## Choosing best models ###################################
models = []
for i in range(10):
    models.append(TRAINING.action_unit + '_Model_{}'.format(i+1))

for i in range(len(models)):
    model_filename = 'models/' + TRAINING.action_unit + '/' + models[i]
    print(i)
    history = create_model_from_best_params(x_train, y_train, x_val, y_val, TRAINING.best_params, DATASET.mapping_AU_to_Index[TRAINING.action_unit], model_filename, i)
    evaluate_model(model_filename, history, x_test, y_test)

print_evaluations(models)