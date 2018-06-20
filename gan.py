from __future__ import print_function, division



from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
import util



import sys



import numpy as np


class GAN():

    def __init__(self):


        self.video_feature_dim = 1024
        self.audio_feature_dim = 1140

        optimizer = Adam(0.0002,0.5)
        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',

            optimizer=SGD(),

            metrics=['accuracy'])


        self.generator = self.build_generator()

        #generator input: g_input (video_feature) & output: g_output (audio_feature)
        g_input = Input(shape=(self.video_feature_dim,))
        g_output = self.generator(g_input)

        #discriminator input: g_output (audio_feature) & ouptut: validity
        self.discriminator.trainable = False
        validity = self.discriminator(g_output)

        #The combine model
        self.combine = Model(g_input, validity)
        self.combine.compile(loss='binary_crossentropy', optimizer=optimizer)
        

    def build_generator(self):



        model = Sequential()



        model.add(Dense(512, input_dim=self.video_feature_dim))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation('leaky_relu'))
        #model.add(Dropout(0.2))


        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization(momentum=0.8))
        #model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        #model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(1140))
        model.add(Activation('tanh'))



        model.summary()



        video_feature = Input(shape=(self.video_feature_dim,))

        result_video = model(video_feature)



        return Model(video_feature, result_video)


    def build_discriminator(self):



        model = Sequential()



        #model.add(Dense(1024,input_dim=self.audio_feature_dim))

        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.5, input_shape = (self.audio_feature_dim,)))
        model.add(Dense(256, input_dim = self.audio_feature_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        #model.add(Dense(256))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))


        #model.add(Dense(64))
        #model.add(BatchNormalization(momentum=0.8))
        #model.add(LeakyReLU(alpha=0.2))
        #model.add(Dropout(0.25))


        #model.add(Dense(64))
        #model.add(LeakyReLU(alpha=0.4))
        #model.add(Dropout(0.5, input_shape = (64,)))


        model.add(Dense(1,activation='sigmoid'))
        model.summary()



        audio_feature = Input(shape=(self.audio_feature_dim,))

        validity = model(audio_feature)


        return Model(audio_feature, validity)
    
    def train(self, epochs=10000, batch_size=128):

        #training features
        features = util.get_feature('../final_video_features_with_scene_train/','../audio_features_train/','../youtube_id.txt')
        video_features = np.array(features.load_video_features())
        audio_features = np.array(features.load_audio_features())

        # Adversarial ground truths

        valid = np.ones((batch_size, 1))

        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            
            #training discriminator
            idx = np.random.randint(0, len(audio_features), batch_size)
            audio_feature = audio_features[idx]

            video_feature = video_features[idx]

            #generate audio features from generator
            gen_audio_feature = self.generator.predict(video_feature)

            #discriminator loss
            d_loss_real = self.discriminator.train_on_batch(audio_feature, valid)

            d_loss_fake = self.discriminator.train_on_batch(gen_audio_feature, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #train generator
            g_loss = self.combine.train_on_batch(video_feature, valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
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

        save(self.generator, "gan_generator4")

        
if __name__ == '__main__':

    gan = GAN()
    gan.train(epochs=4000, batch_size=100)
    #gan.save_model()
