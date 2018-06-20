import os
import pickle
import numpy as np

class get_feature():
    
    def __init__(self,video_feature_path,audio_feature_path,youtube_id_dir):

        self.video_feature_path = video_feature_path
        self.audio_feature_path = audio_feature_path
        self.name_list = list()
        for root, dirs, files in os.walk(video_feature_path):
            for f in files:
                name = f.split(".")[0]
                self.name_list.append(name)

        self.dir_list = list()
        with open(youtube_id_dir,'r') as f:
            for x in f.readlines():
                x = x.strip()
                if x in self.name_list:
                    self.dir_list.append(x)

    def load_video_features(self):
        all_features = list()
        for name in self.dir_list:
            path = (self.video_feature_path + name +'.pickle')
            feature = pickle.load(open(path,'rb'))
            #feature = feature / max(feature)
            #print (max(feature))
            #tmp = np.mean(feature,axis=0)
            all_features.append(feature)
        return all_features

    def load_audio_features(self):
        all_features = list()
        for name in self.dir_list:
            path = (self.audio_feature_path + name +'.pkl')
            feature = pickle.load(open(path,'rb'))
            all_features.append(feature)
        return all_features
if __name__ == '__main__':
    features = get_feature('../final_video_features_with_scene_train/','../test_audio_features/','../youtube_id.txt')
    video_features = features.load_video_features()
    #audio_features = features.load_audio_features()
    


