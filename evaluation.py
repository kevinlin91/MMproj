from keras.models import load_model
from util import get_feature
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from keras.models import model_from_json
from sklearn.metrics.pairwise import cosine_similarity as cs
import wgan
import gan
import os
import pickle
import copy

class evaluation():
    def __init__(self):
        self.features = get_feature('./final_video_features_with_scene_test/', './audio_features_test/', './youtube_id.txt')
        self.video_features = self.features.load_video_features()
        self.audio_features = self.features.load_audio_features()
        self.dataset_length = len(self.audio_features)
        self.library_features, self.library_list = self.load_library_features()
        self.library_length = len(self.library_features)
        self.cfa_video = pickle.load(open('./models/saved_models/cfa_video.pickle','rb'))
        self.cfa_audio = pickle.load(open('./models/saved_models/cfa_audio.pickle','rb'))
        self.cfa_video_features = np.dot(np.array([ x.tolist() for x in self.video_features]), self.cfa_video)
        self.cfa_audio_features = np.dot(np.array([ x.tolist() for x in self.audio_features]), self.cfa_audio)
        self.cfa_library_features = np.dot(np.array([ x.tolist() for x in self.library_features]), self.cfa_audio)
        
    def load_library_features(self):
        library_list = list()
        for root, dirs, files in os.walk('./music_library_features'):
            library_list = files
        library_features = list()
        for f in files:
            path = './music_library_features/' + f
            feature = pickle.load(open(path,'rb'))
            library_features.append(feature)
        return library_features, library_list
        
    def wgan_evaluation(self):
        generator = wgan.WGAN().build_generator()
        generator.load_weights('./models/saved_models/wgan_generator3_weights.hdf5')
        fake_audio_features = generator.predict_on_batch(np.array(self.video_features))
        #result = self.score(fake_audio_features)
        result = self.score_library(fake_audio_features)
        return result
    def gan_evaluation(self):
        #generator = gan.GAN().build_generator()
        json_file = open('./models/saved_models/gan_generator4.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        generator = model_from_json(loaded_model_json)
        generator.load_weights('./models/saved_models/gan_generator4_weights.hdf5')
        fake_audio_features = generator.predict_on_batch(np.array(self.video_features))
        #result = self.score(fake_audio_features)
        result = self.score_library(fake_audio_features)
        return result
    def cfa_evaluation(self):
        #result = self.score_cfa(self.cfa_video_features)
        result = self.score_library_cfa(self.cfa_video_features)
        return result
    def score(self, fake_audio_features):
        total_score = 0.0
        for index_video, fake_audio_feature in enumerate(fake_audio_features):
            similarity = list()
            for index_audio, audio_feature in enumerate(self.audio_features):
                fake_audio_feature = np.array(fake_audio_feature).reshape(1,-1)
                audio_feature = np.array(audio_feature).reshape(1,-1)
                sim = cs(fake_audio_feature, audio_feature).tolist()
                similarity +=sim[0]
            ranking = np.argsort(similarity)
            total_score += float(ranking[index_video]) / float(self.dataset_length)
        return total_score / float(self.dataset_length)
    def score_library(self, fake_audio_features):
        total_score = 0.0
        for index_video, fake_audio_feature in enumerate(fake_audio_features):
            similarity = list()
            #print (len(self.library_features))
            tmp_library_features = copy.deepcopy(self.library_features)
            tmp_library_features.append(self.audio_features[index_video])
            for index_audio, audio_feature in enumerate(tmp_library_features):
                fake_audio_feature = np.array(fake_audio_feature).reshape(1,-1)
                audio_feature = np.array(audio_feature).reshape(1,-1)
                sim = cs(fake_audio_feature, audio_feature).tolist()
                similarity +=sim[0]
            ranking = np.argsort(similarity)
            total_score += float(ranking[-1]) / float(self.library_length)
        return total_score / float(self.dataset_length)
    def score_cfa(self, fake_audio_features):
        total_score = 0.0
        for index_video, fake_audio_feature in enumerate(fake_audio_features):
            similarity = list()
            for index_audio, audio_feature in enumerate(self.cfa_audio_features):
                fake_audio_feature = np.array(fake_audio_feature).reshape(1,-1)
                audio_feature = np.array(audio_feature).reshape(1,-1)
                sim = cs(fake_audio_feature, audio_feature).tolist()
                similarity +=sim[0]
            ranking = np.argsort(similarity)
            total_score += float(ranking[index_video]) / float(self.dataset_length)
        return total_score / float(self.dataset_length)
    def score_library_cfa(self, fake_audio_features):
        total_score = 0.0
        for index_video, fake_audio_feature in enumerate(fake_audio_features):
            similarity = list()
            #print (len(self.library_features))
            tmp_library_features = copy.deepcopy(self.cfa_library_features)
            new_tmp_library_features= tmp_library_features.tolist().append(self.cfa_audio_features[index_video].tolist())
            for index_audio, audio_feature in enumerate(tmp_library_features):
                fake_audio_feature = np.array(fake_audio_feature).reshape(1,-1)
                audio_feature = np.array(audio_feature).reshape(1,-1)
                sim = cs(fake_audio_feature, audio_feature).tolist()
                similarity +=sim[0]
            ranking = np.argsort(similarity)
            total_score += float(ranking[-1]) / float(self.library_length)
        return total_score / float(self.dataset_length)
if __name__ == '__main__':
    EVA = evaluation()
    result = EVA.gan_evaluation()
    print (result)
        

        



    
    
                     
