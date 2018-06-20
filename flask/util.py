import os
import pickle
import numpy as np

class get_feature():
    
    def __init__(self,video_feature_path,audio_feature_path):

        self.video_feature_path = video_feature_path
        self.audio_feature_path = audio_feature_path
        self.video_list = list()
        for root, dirs, files in os.walk(video_feature_path):
            for f in files:
                name = f.split(".")[0]
                self.video_list.append(name)

        self.audio_list = list()
        for root, dirs, files in os.walk(audio_feature_path):
            for f in files:
                name = f.split(".")[0]
                self.audio_list.append(name)

    def load_video_features(self):
        all_features = list()
        for name in self.video_list:
            path = (self.video_feature_path + name +'.pickle')
            feature = pickle.load(open(path,'rb'))
            feature = feature / max(feature)
            #print (max(feature))
            #tmp = np.mean(feature,axis=0)
            all_features.append(feature)
        return all_features

    def load_audio_features(self):
        all_features = list()
        for name in self.audio_list:
            path = (self.audio_feature_path + name +'.pkl')
            feature = pickle.load(open(path,'rb'))
            all_features.append(feature)
        return all_features,self.audio_list
if __name__ == '__main__':
    features = get_feature('../final_video_features_with_scene_train/','../test_audio_features/','../youtube_id.txt')
    video_features = features.load_video_features()
    #audio_features = features.load_audio_features()
    


