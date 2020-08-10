class Dataset:
# Comment and uncomment the relevant AU datasets that need to be used for data_organising.py

    #CK+
#    data_source = 'CK+'
#    images_path = "\data\CKPlusDataset\cohn-kanade-images"
#    labels_path = "\data\CKPlusDataset\FACS"
#    emotion_labels_path = "\data\CKPlusDataset\Emotion"
#    number_of_AUs = 43
        
    #DISFA
    data_source = 'DISFA'
    images_path = "\data\DISFA\Frames_RightCamera"
    labels_path = "\data\DISFA\ActionUnit_Labels"
    number_of_AUs = 12
    
    #DISFA+
#    data_source = 'DISFA+'
#    images_path = "\data\DISFAPlusDataset\Images"
#    labels_path = "\data\DISFAPlusDataset\Labels"
#    number_of_AUs = 12
    
    mapping_CK_to_DISFA = {0:0, 4:1, 8:3, 9:4, 10:5, 1:11,
                           2:14, 3:16, 5:19, 6:24, 7:25, 11:8}
    mapping_AU_to_Index = {"AU1":0, "AU12":1, "AU15":2, "AU17":3,
                           "AU2":4, "AU20":5, "AU25":6, "AU26":7,
                           "AU4":8, "AU5":9, "AU6":10, "AU9":11}
    #0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
    mapping_emotion_number_to_valence = {0:0, 1:-0.40, 2:-0.58, 3:-0.68, 4:-0.12, 5:0.89, 6:-0.81, 7:0.18}
    mapping_emotion_number_to_arousal = {0:0, 1:0.79, 2:0.66, 3:0.49, 4:0.79, 5:0.17, 6:-0.4, 7:0.69}
    upper_facial_AUs = {0, 4, 8, 9}
    lower_facial_AUs = {1, 2, 3, 5, 6, 7}
    whole_facial_AUs = {10, 11}
    face_vertical_size = 224
    face_horizontal_size = 160
    upper_face_vertical_size = 144
    lower_face_vertical_size = 96
    model_file = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_file = "deploy.prototxt"
    min_confidence = 0.8

class Training:
    test_size = 0.2
    val_size = 0.25
    action_unit = "AU1"
    dataset = ['DISFA_40_0', 'DISFA+_40_0', 'DISFA_50_5-3', 'DISFA+_50_5-3'] #General dataset
#    dataset = ['DISFA_40_0', 'DISFA+_40_0', 'DISFA_50_5-3', 'DISFA+_50_5-3', 'DISFA+_10_0_6_4', 'DISFA_10_0_6_4'] #AU4 dataset
#    dataset = ['DISFA_30_0', 'DISFA+_30_0', 'DISFA_50_5-3', 'DISFA+_50_5-3'] #AU5 and AU15 dataset
#    dataset = ['DISFA_40_0', 'DISFA+_40_0', 'DISFA_50_5-3', 'DISFA+_50_5-3', 'DISFA+_10_0_4_9', 'DISFA+_10_0_6_9', 'DISFA_10_0_4_9', 'DISFA_10_0_6_9' ] #AU9 dataset
    
    #First, second and third params are the same for all action units
    first_params = {'conv_layers':[2, 3],
                       'feature_maps_1':[8, 16, 32],
                       'fully_connected_layers':[1, 2],
                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
                       'batch_before_after':['before', 'after'],
                       'batch_norm_1':['bn1_0', 'bn1_1'],
                       'batch_norm_2':['bn2_0', 'bn2_1'],
                       'batch_norm_3':['bn3_0', 'bn3_1'],
                       'batch_norm_4':['bn4_0', 'bn4_1'],
                       'batch_norm_5':['bn5_0', 'bn5_1'],
                       'batch_size':[64],
                       'optimizer':['Adam'],
                       'learning_rate': [0.001],
                       'momentum': [0],
                       'reduceLR':[0],
                       'epochs': [8]}

    second_params = {'conv_layers':[2, 3],
                       'feature_maps_1':[8, 16, 32],
                       'fully_connected_layers':[1, 2],
                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
                       'batch_before_after':['before', 'after'],
                       'batch_norm_1':['bn1_0', 'bn1_1'],
                       'batch_norm_2':['bn2_0', 'bn2_1'],
                       'batch_norm_3':['bn3_0', 'bn3_1'],
                       'batch_norm_4':['bn4_1'],
                       'batch_norm_5':['bn5_0', 'bn5_1'],
                       'batch_size':[64],
                       'optimizer':['Adam'],
                       'learning_rate': [0.001],
                       'momentum': [0],
                       'reduceLR':[0],
                       'epochs': [8]}

    third_params = {'conv_layers':[2, 3],
                       'feature_maps_1':[8, 16, 32],
                       'fully_connected_layers':[2],
                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
                       'batch_before_after':['before'],
                       'batch_norm_1':['bn1_1'],
                       'batch_norm_2':['bn2_0', 'bn2_1'],
                       'batch_norm_3':['bn3_0', 'bn3_1'],
                       'batch_norm_4':['bn4_1'],
                       'batch_norm_5':['bn5_0', 'bn5_1'],
                       'batch_size':[64],
                       'optimizer':['Adam'],
                       'learning_rate': [0.001],
                       'momentum': [0],
                       'reduceLR':[0],
                       'epochs': [8]}
    
    # AU1 - fourth, opt, mom and best params
    fourth_params = {'conv_layers':[3],
                       'feature_maps_1':[8, 16, 32],
                       'fully_connected_layers':[2],
                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
                       'batch_before_after':['before'],
                       'batch_norm_1':['bn1_1'],
                       'batch_norm_2':['bn2_1'],
                       'batch_norm_3':['bn3_1'],
                       'batch_norm_4':['bn4_1'],
                       'batch_norm_5':['bn5_0'],
                       'batch_size':[32, 64, 128, 256],
                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
                       'momentum': [0, 0.9, 0.95, 0.99],
                       'reduceLR':[1],
                       'epochs': [15]}

    opt_params = {'conv_layers':[3],
                       'feature_maps_1':[8, 16, 32],
                       'fully_connected_layers':[2],
                       'fc_neurons_1':['fc1_128', 'fc1_256', 'fc1_512'],
                       'fc_neurons_2':['fc2_128', 'fc2_256', 'fc2_512'],
                       'batch_before_after':['before'],
                       'batch_norm_1':['bn1_1'],
                       'batch_norm_2':['bn2_1'],
                       'batch_norm_3':['bn3_1'],
                       'batch_norm_4':['bn4_1'],
                       'batch_norm_5':['bn5_0'],
                       'batch_size':[32],
                       'optimizer':['Adam', 'RMSprop'],
                       'learning_rate': [0.001],
                       'momentum': [0],
                       'reduceLR':[1],
                       'epochs': [15]}

    mom_params = {'conv_layers':[3],
                       'feature_maps_1':[8, 16, 32],
                       'fully_connected_layers':[2],
                       'fc_neurons_1':['fc1_128', 'fc1_256', 'fc1_512'],
                       'fc_neurons_2':['fc2_128', 'fc2_256', 'fc2_512'],
                       'batch_before_after':['before'],
                       'batch_norm_1':['bn1_1'],
                       'batch_norm_2':['bn2_1'],
                       'batch_norm_3':['bn3_1'],
                       'batch_norm_4':['bn4_1'],
                       'batch_norm_5':['bn5_0'],
                       'batch_size':[32],
                       'optimizer':['SGD+N'],
                       'learning_rate': [0.001],
                       'momentum': [0.9, 0.95, 0.99],
                       'reduceLR':[1],
                       'epochs': [15]}
    
    best_params = {'feature_maps_1': [16, 32, 8, 16, 32, 8, 16, 8, 32, 8],
                   'fc_neurons_1':[128, 512, 256, 256, 128, 256, 256, 128, 256, 256],
                   'fc_neurons_2':[128, 128, 128, 128, 512, 128, 128, 128, 256, 512],
                   'optimizer':['RMSprop', 'Adam', 'SGD+N', 'SGD+N', 'Adam', 'RMSprop', 'RMSprop', 'RMSprop', 'SGD+N', 'Adam'],
                   'momentum': [0, 0, 0.95, 0.95, 0, 0, 0, 0, 0.9, 0]}
    
    # AU2 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    best_params = {'feature_maps_1': [8, 8, 16, 32, 32, 8, 16, 16, 32, 8],
#              'fc_neurons_1':[512, 256, 128, 512, 512, 128, 512, 128, 128, 256],
#              'fc_neurons_2':[128, 256, 512, 256, 512, 128, 256, 256, 128, 512],
#              'optimizer':['SGD+N', 'Adam', 'Adam', 'RMSprop', 'SGD+N', 'Adam', 'RMSprop', 'SGD+N', 'SGD+N', 'SGD+N'],
#              'momentum': [0.9, 0, 0, 0, 0.95, 0, 0, 0.95, 0.95, 0.95]}
    
    # AU4 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#    
#    best_params = {'feature_maps_1': [16, 8, 16, 8, 32, 32, 16, 32, 32, 8],
#              'fc_neurons_1':[128, 512, 512, 128, 128, 256, 256, 512, 128, 512],
#              'fc_neurons_2':[512, 512, 128, 256, 256, 512, 256, 256, 512, 512],
#              'optimizer':['Adam', 'SGD+N', 'SGD+N', 'SGD+N', 'SGD+N', 'Adam', 'SGD+N', 'RMSprop', 'SGD+N', 'SGD+N'],
#              'momentum': [0, 0.95, 0.95, 0.9, 0.99, 0, 0.95, 0, 0.9, 0.99]}
    
    # AU5 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop', 'SGD'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#    
#    best_params = {'feature_maps_1': [16, 16, 32, 16, 8, 32, 32, 8, 32, 16],
#              'fc_neurons_1':[64, 256, 512, 256, 256, 128, 64, 512, 256, 512],
#              'fc_neurons_2':[256, 64, 128, 512, 64, 512, 256, 512, 64, 128],
#              'optimizer':['Adam', 'RMSprop', 'SGD+N', 'SGD+N', 'Adam', 'Adam', 'RMSprop', 'SGD+N', 'SGD+N', 'RMSprop'],
#              'momentum': [0, 0, 0.95, 0.95, 0, 0, 0, 0.95, 0.95, 0]}
#    
#    # AU6 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop', 'SGD'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#    
#    best_params = {'feature_maps_1': [16, 16, 32, 32, 16, 32, 16, 8, 8, 32],
#              'fc_neurons_1':[128, 256, 256, 128, 128, 64, 64, 64, 256, 64],
#              'fc_neurons_2':[64, 128, 64, 64, 64, 256, 256, 128, 256, 64],
#              'optimizer':['SGD+N', 'Adam', 'RMSprop', 'Adam', 'Adam', 'Adam', 'RMSprop', 'RMSprop', 'RMSprop', 'RMSprop'],
#              'momentum': [0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
#    
#    # AU9 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop', 'SGD'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#    
#    
#    best_params = {'feature_maps_1': [8, 32, 8, 32, 8, 32, 16, 16, 32, 16],
#              'fc_neurons_1':[256, 256, 256, 64, 128, 64, 128, 64, 256, 128],
#              'fc_neurons_2':[128, 512, 256, 64, 512, 256, 256, 128, 64, 128],
#              'optimizer':['RMSprop', 'Adam', 'RMSprop', 'Adam', 'Adam', 'RMSprop', 'Adam', 'SGD+N', 'SGD+N', 'SGD+N'],
#              'momentum': [0, 0, 0, 0, 0, 0, 0, 0.95, 0.9, 0.9]}
#    
#    # AU12 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    best_params = {'feature_maps_1': [32, 16, 32, 8, 32, 8, 16, 32, 32, 8],
#              'fc_neurons_1':[64, 512, 128, 256, 128, 512, 64, 128, 512, 64],
#              'fc_neurons_2':[128, 128, 128, 128, 64, 256, 64, 256, 256, 128],
#              'optimizer':['RMSprop', 'Adam', 'SGD+N', 'SGD+N', 'RMSprop', 'SGD+N', 'SGD+N', 'Adam', 'Adam', 'RMSprop'],
#              'momentum': [0, 0, 0.95, 0.95, 0, 0.9, 0.99, 0, 0, 0]}
#    
#    # AU15 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_0'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256'],
#                       'fc_neurons_2':['fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_0'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256'],
#                       'fc_neurons_2':['fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_0'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    best_params = {'feature_maps_1': [32, 32, 16, 16, 16, 32, 16, 16, 16, 16],
#              'fc_neurons_1':[256, 128, 128, 64, 128, 64, 64, 256, 128, 128],
#              'fc_neurons_2':[128, 256, 512, 128, 128, 128, 512, 128, 256, 128],
#              'optimizer':['Adam', 'SGD+N', 'RMSprop', 'RMSprop', 'Adam', 'Adam', 'RMSprop', 'RMSprop', 'Adam', 'RMSprop'],
#              'momentum': [0, 0.9, 0, 0, 0, 0, 0, 0, 0, 0]}
#
#    # AU17 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    best_params = {'feature_maps_1': [16, 8, 8, 16, 16, 16, 8, 32, 8, 16],
#              'fc_neurons_1':[64, 512, 128, 512, 256, 256, 128, 512, 256, 512],
#              'fc_neurons_2':[64, 256, 256, 128, 64, 64, 64, 64, 128, 256],
#              'optimizer':['SGD+N', 'SGD+N', 'Adam', 'SGD+N', 'SGD+N', 'Adam', 'SGD+N', 'SGD+N', 'Adam', 'SGD+N'],
#              'momentum': [0.95, 0.95, 0, 0.9, 0.95, 0, 0.9, 0.95, 0, 0.9]}
#    
#    # AU20 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    best_params = {'feature_maps_1': [16, 8, 8, 32, 32, 16, 32, 32, 32, 16],
#               'fc_neurons_1':[64, 64, 128, 64, 64, 128, 256, 64, 128, 64],
#               'fc_neurons_2':[128, 256, 64, 64, 256, 64, 64, 256, 256, 64],
#               'optimizer':['RMSprop', 'Adam', 'RMSprop', 'RMSprop', 'SGD+N', 'RMSprop', 'SGD+N', 'RMSprop','SGD+N', 'Adam'],
#               'momentum': [0, 0, 0, 0, 0.95, 0, 0.95, 0, 0.95, 0]}
#    
#    # AU25 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_1'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_0'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#    
#    best_params = {'feature_maps_1': [32, 32, 32, 16, 32, 32, 32, 16, 16, 16],
#              'fc_neurons_1':[128, 128, 256, 512, 64, 512, 64, 64, 128, 256],
#              'fc_neurons_2':[256, 512, 512, 256, 64, 256, 128, 128, 256, 128],
#              'optimizer':['RMSprop', 'RMSprop', 'RMSprop', 'Adam', 'RMSprop', 'RMSprop', 'RMSprop', 'RMSprop', 'RMSprop', 'RMSprop'],
#              'momentum': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
#    
#    # AU26 - fourth, opt, mom and best params
#    fourth_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_64', 'fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_0'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32, 64, 128, 256],
#                       'optimizer':['Adam', 'SGD', 'SGD+N', 'RMSprop'],
#                       'learning_rate': [0.0001, 0.00055, 0.001, 0.0055, 0.01],
#                       'momentum': [0, 0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    opt_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_0'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['Adam', 'RMSprop'],
#                       'learning_rate': [0.001],
#                       'momentum': [0],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#
#    mom_params = {'conv_layers':[3],
#                       'feature_maps_1':[8, 16, 32],
#                       'fully_connected_layers':[2],
#                       'fc_neurons_1':['fc1_128', 'fc1_256', 'fc1_512'],
#                       'fc_neurons_2':['fc2_64', 'fc2_128', 'fc2_256', 'fc2_512'],
#                       'batch_before_after':['before'],
#                       'batch_norm_1':['bn1_1'],
#                       'batch_norm_2':['bn2_1'],
#                       'batch_norm_3':['bn3_0'],
#                       'batch_norm_4':['bn4_1'],
#                       'batch_norm_5':['bn5_1'],
#                       'batch_size':[32],
#                       'optimizer':['SGD+N'],
#                       'learning_rate': [0.001],
#                       'momentum': [0.9, 0.95, 0.99],
#                       'reduceLR':[1],
#                       'epochs': [15]}
#    
#    best_params = {'feature_maps_1': [8, 8, 32, 16, 32, 8, 8, 8, 32, 8],
#              'fc_neurons_1':[512, 256, 128, 512, 256, 512, 128, 512, 128, 256],
#              'fc_neurons_2':[256, 128, 256, 256, 256, 128, 256, 256, 64, 64],
#              'optimizer':['SGD+N', 'RMSprop', 'SGD+N', 'SGD+N', 'SGD+N', 'RMSprop', 'SGD+N', 'Adam', 'SGD+N', 'SGD+N'],
#              'momentum': [0.9, 0, 0.95, 0.95, 0.9, 0, 0.95, 0, 0.9, 0.95]}
    
class VideoPredictor:
    face_landmarks_file = "shape_predictor_68_face_landmarks.dat"
    ear_threshold = 0.23
    ear_consecutive_frames = 35
    mapping_video_index_to_AU_label_text = {0:"AU1 - Inner Brow Raiser:", 
                                            1:"AU2 - Outer Brow Raiser:", 
                                            2:"AU4 - Brow Lowerer:", 
                                            3:"AU5 - Upper Lid Raiser:",
                                            4:"AU6 - Cheek Raiser:", 
                                            5:"AU9 - Nose Wrinkler:", 
                                            6:"AU12 - Lip Corner Puller:", 
                                            7:"AU15 - Lip Corner Depressor:", 
                                            8:"AU25 - Lips Part:", 
                                            9:"AU26 - Jaw Drop:",
                                            10: "EAR ratio:"}

class Other:
    random_state = 0

DATASET = Dataset()
TRAINING = Training()
VIDEO_PREDICTOR = VideoPredictor()
OTHER = Other()