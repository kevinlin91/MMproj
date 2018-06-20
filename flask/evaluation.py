from keras.models import load_model
from util import get_feature
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from keras.models import model_from_json
from sklearn.metrics.pairwise import cosine_similarity as cs
import wgan
import os
import keras.backend as K


class evaluation():
    def __init__(self):
        K.clear_session()
        self.features = get_feature('./video_features/', './audio_features/')
        self.video_features = self.features.load_video_features()
        self.audio_features, self.audio_list = self.features.load_audio_features()
        self.dataset_length = len(self.audio_features)
        self.generator = wgan.WGAN().build_generator()
        self.generator.load_weights('./models/wgan_generator1_weights.hdf5')
    def wgan_evaluation(self):
        fake_audio_features = self.generator.predict_on_batch(np.array(self.video_features))
        K.clear_session()
        result_ranking = self.score(fake_audio_features)
        #del generator
        return self.audio_list[result_ranking]
    def gan_evaluation(self):
        K.clear_session()
        json_file = open('./models/gan_generator1.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        generator = model_from_json(loaded_model_json)
        generator.load_weights('./models/gan_generator1_weights.hdf5')
        fake_audio_features = generator.predict_on_batch(np.array(self.video_features))
        result_ranking = self.score(fake_audio_features)
        return self.audio_list[result_ranking]
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
        return ranking.tolist().index(0)

if __name__ == '__main__':
    EVA = evaluation()
    wgan_result = EVA.wgan_evaluation()
    with open('./recommend_result.txt','w') as f:
        f.write(wgan_result)
        

        



    
    
                     
