import os
import sys
from feature_extractor import *
from PIL import Image
import numpy as np
import cv2
import pickle
import glob

name = ''
with open('./file_name.txt','r') as f:
    name = f.read()
    
scene = pickle.load(open('./scene_detection_pickle/%s.pickle' % (name.split('.')[0]),'rb'))
extractor = YouTube8MFeatureExtractor()
files = glob.glob('./video_features/*')

for f in files:

    os.remove(f)

    
if scene[0] == -1:
    cap = cv2.VideoCapture('./user_video/%s.mp4' % (name.split('.')[0]))    
    features = list()
    success = True
    while success:
        success, image = cap.read()
        if(success == False):
            continue
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        feature = extractor.extract_rgb_frame_features(image)
        features.append(feature)
    features = np.mean(np.array(features), axis=0)
    pickle.dump(features, open('./video_features/%s.pickle' % (name.split('.')[0]),'wb'))
    
else:
    cap = cv2.VideoCapture('./user_video/%s.mp4' % name.split('.')[0])    
    features = list()
    success = True
    count = 0
    while success:
        success, image = cap.read()
        if(success == False):
            continue
        if count in scene:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            feature = extractor.extract_rgb_frame_features(image)
            features.append(feature)
        count +=1
    features = np.mean(np.array(features), axis=0)
    print (len(features))
    pickle.dump(features, open('./video_features/%s.pickle' % (name.split('.')[0]) ,'wb'))
    



    
