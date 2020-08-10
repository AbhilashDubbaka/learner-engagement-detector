import numpy as np
import random as rn
from parameters import TRAINING, OTHER
np.random.seed(OTHER.random_state)
rn.seed(OTHER.random_state)
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.externals import joblib
from sklearn import preprocessing
import sys

#%matplotlib inline

def import_and_extract_train_test_for_AU(dataset, output):
    print("Importing data...")
    if output == "arousal":
        df = pd.read_excel(dataset, "CK+_processed")
        df = df.iloc[:,1:]
        df = df.dropna()
        df2 = pd.read_excel(dataset, "KDEF_processed")
        df2 = df2.iloc[:,1:]
        df = pd.concat([df, df2])
        df = df.reset_index(drop = True)
        df2 = pd.read_excel(dataset, "Radboud_processed")
        df2 = df2.iloc[:,1:]
        df = pd.concat([df, df2])
        df = df.reset_index(drop = True)
    
    else:
        df = pd.read_excel(dataset, "Radboud_processed")
        df = df.iloc[:,1:]
    
    #Return a random sample of 1 i.e. 100% after shuffling rows
    df = df.sample(frac = 1, random_state = OTHER.random_state).reset_index(drop=True)

    print("Creating x and y data groups...")
    data = df.values
    x1 = data[:,0:8] #0:7 for no sad face
    x2 = data[:,8:10]
    x = np.concatenate((x1, x2), axis=1)
    x = x.astype('float32')
    if output == "arousal":
        y = data[:,12]
    else:
        y = data[:,11]
    y = y.astype('float32')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = TRAINING.test_size, random_state = OTHER.random_state)
    return x_train, x_test, y_train, y_test

def create_svr_grid_search(x_train, y_train, param_grid, K, kernel_type):
    old_stdout = sys.stdout
    log_file = open("message_" + kernel_type + ".log", "w")
    sys.stdout = log_file
    svr = SVR()
    scaler = preprocessing.Normalizer()
    x_train = scaler.fit_transform(x_train)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    clf = GridSearchCV(svr, param_grid, cv = K, verbose=100, scoring=scorer, n_jobs=-1)
    sys.stdout = old_stdout
    log_file.close()
    log_file = open("message_" + kernel_type + ".log", "w")
    sys.stdout = log_file
    print("Starting Grid Search")
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    sys.stdout = old_stdout
    log_file.close()
    joblib.dump(clf, "clf_object_" + kernel_type + ".pkl")

def create_svr(x_train, y_train, model_params):    
    if model_params['kernel'] == 'linear':
        svr = SVR(C=model_params['C'], epsilon=model_params['epsilon'], kernel=model_params['kernel'])
    elif model_params['kernel'] == 'rbf' or model_params['kernel'] == 'sigmoid':
        svr = SVR(C=model_params['C'], epsilon=model_params['epsilon'], gamma=model_params['gamma'], kernel=model_params['kernel'])
    else:
        svr = SVR(C=model_params['C'], degree=model_params['degree'], epsilon=model_params['epsilon'], gamma=model_params['gamma'], kernel=model_params['kernel'])
    
    scaler = preprocessing.Normalizer()
    x_train = scaler.fit_transform(x_train)
    
    scores = cross_val_score(svr, x_train, y_train, cv=10)
    print(np.mean(scores))
    
    svr.fit(x_train, y_train)
    return svr, scaler

#################################################################################
# Main functions
dataset = "AU_to_Val_Ars.xlsx"

#Valence
x_train, x_test, y_train, y_test = import_and_extract_train_test_for_AU(dataset, "valence")
#
##Experiment 1
#K = 3
#param_grid_linear = [
#   {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']}
#]
#
#param_grid_sigmoid = [
#   {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['sigmoid']}
#]
#
#param_grid_rbf = [
#   {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf']}
#]
#
#param_grid_poly = [
#   {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 'degree': [3], 'kernel': ['poly']} #decided not to fine tune degree straight away
#]
#
#print("Starting experiement 1...")
#create_svr_grid_search(x_train, y_train, param_grid_linear, K, "val_linear_1")
#create_svr_grid_search(x_train, y_train, param_grid_sigmoid, K, "val_sigmoid_1")
#create_svr_grid_search(x_train, y_train, param_grid_rbf, K, "val_rbf_1")
#create_svr_grid_search(x_train, y_train, param_grid_poly, K, "val_poly_1")
#
#Experiment 2
#K = 3
#param_grid_linear = [
#   {'C': [0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 'kernel': ['linear']}
#]
#
#param_grid_sigmoid = [
#   {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], 'kernel': ['sigmoid']} 
#]
#
#param_grid_rbf = [
#   {'C': [0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 0.5], 'gamma': [0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'kernel': ['rbf']} 
#]
#
#param_grid_poly = [
#   {'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 100, 200], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], 'degree': [2, 3, 4], 'kernel': ['poly']}
#]
#
#print("Starting experiement 2...")
#create_svr_grid_search(x_train, y_train, param_grid_linear, K, "val_linear_2")
#create_svr_grid_search(x_train, y_train, param_grid_sigmoid, K, "val_sigmoid_2")
#create_svr_grid_search(x_train, y_train, param_grid_rbf, K, "val_rbf_2")
#create_svr_grid_search(x_train, y_train, param_grid_poly, K, "val_poly_2_norm")
#
##Experiment 3
#K = 3
#param_grid_linear = [
#   {'C': [0.005, 0.01, 0.05, 0.1, 0.5], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1], 'kernel': ['linear']} #Seems to be stagnating around 0.115
#]
#
#param_grid_sigmoid = [
#   {'C': [50, 75, 100, 125, 150], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1], 'gamma': [0.00005, 0.0001, 0.0005, 0.001], 'kernel': ['sigmoid']} #Tried different values of C since there does seem to be a difference, epsilon doesnt seem to be affecting much so keep as is, gamma around 0.0001
#]
#
#param_grid_rbf = [
#   {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1], 'gamma': [0.01, 0.05, 0.1, 0.5, 1], 'kernel': ['rbf']} #Not great with c= 0.01. Not much in it tbh, seems to be around 0.083 so decided to put more values in ranges and pick best one
#]
#
#print("Starting experiement 3...")
#create_svr_grid_search(x_train, y_train, param_grid_linear, K, "val_linear_3_norm")
#create_svr_grid_search(x_train, y_train, param_grid_sigmoid, K, "val_sigmoid_3_norm")
#create_svr_grid_search(x_train, y_train, param_grid_rbf, K, "val_rbf_3_norm")

#Final CV
#K = 5
#param_grid_linear = [
#   {'C': [0.05], 'epsilon': [0.1], 'kernel': ['linear']}
#]
#
#param_grid_sigmoid = [
#   {'C': [75], 'epsilon': [0.1], 'gamma': [0.001], 'kernel': ['sigmoid']} 
#]
#
#param_grid_rbf = [
#   {'C': [10], 'epsilon': [0.1], 'gamma': [0.5], 'kernel': ['rbf']}
#]
#
#param_grid_poly = [
#   {'C': [1], 'epsilon': [0.1], 'gamma': [0.5], 'degree': [2], 'kernel': ['poly']}
#]

model_params = {'C': 0.5, 'epsilon': 0.1, 'gamma': 1, 'degree':4, 'kernel': 'poly'}
svr_val, scaler = create_svr(x_train, y_train, model_params)
x_test = scaler.transform(x_test)
predictions = svr_val.predict(x_test)
print(svr_val.score(x_test,y_test))
plt.scatter(y_test, predictions, color='black', label='Data')
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.grid()
plt.savefig('models/Valence/scatter_valence.png', dpi=800)
plt.show()

x_train = np.concatenate((x_train, x_test))
y_train = np.concatenate((y_train, y_test))
model_params = {'C': 0.5, 'epsilon': 0.1, 'gamma': 1, 'degree':4, 'kernel': 'poly'}
clf, scaler = create_svr(x_train, y_train, model_params)
joblib.dump(clf, 'models/Valence/valence.pkl')
 
############################################################################################################
#Arousal
x_train, x_test, y_train, y_test = import_and_extract_train_test_for_AU(dataset, "arousal")
#Experiment 1
#K = 3
#param_grid_linear = [
#   {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']}
#]
#
#param_grid_sigmoid = [
#   {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['sigmoid']}
#]
#
#param_grid_rbf = [
#   {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf']}
#]
#
#param_grid_poly = [
#   {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 'degree': [3], 'kernel': ['poly']} #decided not to fine tune degree straight away
#]
#
#print("Starting experiement 1...")
#create_svr_grid_search(x_train, y_train, param_grid_linear, K, "ars_linear_1")
#create_svr_grid_search(x_train, y_train, param_grid_sigmoid, K, "ars_sigmoid_1")
#create_svr_grid_search(x_train, y_train, param_grid_rbf, K, "ars_rbf_1")
#create_svr_grid_search(x_train, y_train, param_grid_poly, K, "ars_poly_1")

#Experiment 2
#K = 3
#param_grid_linear = [
#   {'C': [0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 'kernel': ['linear']} 
#]
#
#param_grid_sigmoid = [
#   {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], 'kernel': ['sigmoid']} 
#]
#
#param_grid_rbf = [
#   {'C': [0.01, 0.1, 1, 10, 100, 1000], 'epsilon': [0.001, 0.01, 0.1, 0.5], 'gamma': [0.01, 0.05, 0.1, 0.5, 1, 2, 5], 'kernel': ['rbf']} 
#]
#
#param_grid_poly = [
#   {'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], 'degree': [2, 3, 4], 'kernel': ['poly']}
#]
#
#print("Starting experiement 2...")
#create_svr_grid_search(x_train, y_train, param_grid_linear, K, "ars_linear_2")
#create_svr_grid_search(x_train, y_train, param_grid_sigmoid, K, "ars_sigmoid_2")
#create_svr_grid_search(x_train, y_train, param_grid_rbf, K, "ars_rbf_2")
#create_svr_grid_search(x_train, y_train, param_grid_poly, K, "ars_poly_2")

#Experiment 3
#K = 3
#param_grid_linear = [
#   {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000], 'epsilon': [0.05, 0.1, 0.15, 0.2, 0.25], 'kernel': ['linear']} 
#]
#
#param_grid_sigmoid = [
#   {'C': [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 'epsilon': [0.05, 0.1, 0.15, 0.25], 'gamma': [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01], 'kernel': ['sigmoid']} 
#]
#
#param_grid_rbf = [
#   {'C': [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000], 'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1], 'gamma': [0.05, 0.1, 0.25, 0.5, 0.75, 1], 'kernel': ['rbf']} 
#]
#
#param_grid_poly = [
#   {'C': [0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 250, 500, 750, 1000], 'epsilon': [0.01, 0.05, 0.1, 0.5], 'gamma': [0.01, 0.05, 0.1, 0.5], 'degree': [2, 3, 4], 'kernel': ['poly']} 
#]
#
#print("Starting experiement 3...")
#create_svr_grid_search(x_train, y_train, param_grid_linear, K, "ars_linear_3_norm")
#create_svr_grid_search(x_train, y_train, param_grid_sigmoid, K, "ars_sigmoid_3_norm")
#create_svr_grid_search(x_train, y_train, param_grid_rbf, K, "ars_rbf_3_norm")
#create_svr_grid_search(x_train, y_train, param_grid_poly, K, "ars_poly_3_norm")

#Final CV
#K = 5
#param_grid_linear = [
#   {'C': [0.05], 'epsilon': [0.15], 'kernel': ['linear']}
#]
#
#param_grid_sigmoid = [
#   {'C': [600], 'epsilon': [0.15], 'gamma': [0.0001], 'kernel': ['sigmoid']}  
#]
#
#param_grid_rbf = [
#   {'C': [1], 'epsilon': [0.1], 'gamma': [0.75], 'kernel': ['rbf']}
#]
#
#param_grid_poly = [
#   {'C': [1000], 'epsilon': [0.1], 'gamma': [0.05], 'degree': [3], 'kernel': ['poly']} 
#]
model_params = {'C': 5, 'epsilon': 0.1, 'gamma': 1, 'kernel': 'rbf'}
svr_val_rbf, scaler = create_svr(x_train, y_train, model_params)
x_test = scaler.transform(x_test)
predictions = svr_val_rbf.predict(x_test)
print(svr_val_rbf.score(x_test,y_test))
plt.scatter(y_test, predictions, color='black', label='Data')
plt.xlabel('Actual')
plt.ylabel('Predictions')
plt.grid()
plt.savefig('models/Arousal/scatter_arousal.png', dpi=800)
plt.show()

x_train = np.concatenate((x_train, x_test))
y_train = np.concatenate((y_train, y_test))
model_params = {'C': 5, 'epsilon': 0.1, 'gamma': 1, 'kernel': 'rbf'}
clf, scaler = create_svr(x_train, y_train, model_params)
joblib.dump(clf, 'models/Arousal/arousal.pkl') 