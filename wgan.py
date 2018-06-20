from __future__ import print_function, division

from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
import keras.backend as K
import sys
import numpy as np
import util

class WGAN():
    def __init__(self):
        
        self.video_feature_dim = 1024
        self.audio_feature_dim = 1140

        #parameter from paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.0002)
        #optimizer = Adam(lr=0.0001)

        #build critic and generator        
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])        
        self.generator = self.build_generator()

        #generate fake audio feature
        video_feature = Input(shape=(self.video_feature_dim,))
        fake_audio_feature = self.generator(video_feature)

        
        self.critic.trainable = False
        valid = self.critic(fake_audio_feature)
        self.combined = Model(video_feature, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        #input layer
        model.add(Dense(512, input_dim=self.video_feature_dim))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization(momentum=0.8))

        #hidden layer 1
        #model.add(Dense(256))
        #model.add(LeakyReLU())
        #model.add(BatchNormalization(momentum=0.8))


        #hidden layer2
        #model.add(Dense(512))
        #model.add(LeakyReLU())
        #model.add(BatchNormalization(momentum=0.8))


        #output layer
        model.add(Dense(1140))
        model.add(Activation("tanh"))


        #model.summary()
        video_feature = Input(shape=(self.video_feature_dim,))
        fake_audio_feature = model(video_feature)

        return Model(video_feature, fake_audio_feature)

    def build_critic(self):

        model = Sequential()

        #input layer
        model.add(Dense(128, input_dim = self.audio_feature_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.01))
        #model.add(Dropout(0.25))

        #hidden layer1
        #model.add(Dense(256,))
        #imodel.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))

        #hidden layer2
        #model.add(Dense(128,))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))

        #hidden layer3
        #model.add(Dense(64,))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))

        #output layer
        model.add(Dense(1,activation='sigmoid'))


        #model.summary()
        audio_feature = Input(shape=(self.audio_feature_dim,))
        validity = model(audio_feature)

        return Model(audio_feature, validity)

    def train(self, epochs, batch_size=128):

        #get training data
        features = util.get_feature('../final_video_features_with_scene_train/','../audio_features_train/','../youtube_id.txt')
        video_features = np.array(features.load_video_features())
        audio_features = np.array(features.load_audio_features())

        #ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))


        for epoch in range(epochs):
            for _ in range(self.n_critic):

                #get random item 
                idx = np.random.randint(0, len(audio_features), batch_size)
                audio_feature = audio_features[idx]
                video_feature = video_features[idx]
                
                #generate fake audio feature
                fake_audio_feature = self.generator.predict(video_feature)

                # train critic
                d_loss_real = self.critic.train_on_batch(audio_feature, valid)
                d_loss_fake = self.critic.train_on_batch(fake_audio_feature, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            g_loss = self.combined.train_on_batch(video_feature, valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, 1 - d_loss[0], 100*d_loss[1], 1 - g_loss[0]))
        self.save_model()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        #save(self.discriminator, "discriminator")

        
if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=10, batch_size=200)
    wgan.generator.save('./test_wgen.h5')
