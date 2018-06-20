from util import get_feature
import numpy as np
import os
import pickle

file_name = list()
for root, dirs, files in os.walk('./final_video_features_with_scene_train'):
    file_name = files
features = list()
for name in file_name:
    path = './final_video_features_with_scene_train/' + name
    feature = pickle.load(open(path,'rb'))
    features.append(feature)
video_features = np.array(features)
nan_list = list()
for index, i in enumerate(video_features):
    if (np.isnan(i).any())==True:
        nan_list.append(index)

for nan_index in nan_list:
    print (file_name[nan_index])
        
