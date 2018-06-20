from util import get_feature
import numpy as np
import pickle
training_features = get_feature('./final_video_features_with_scene_train/', './audio_features_train/', './youtube_id.txt')
video_features = training_features.load_video_features()
audio_features = training_features.load_audio_features()
np_video_features = np.array([ x.tolist() for x in video_features ]).transpose()
np_audio_features = np.array([ x.tolist() for x in audio_features ])

print (np_video_features.shape)
print (np_audio_features.shape)

input_features = np.dot(np_video_features, np_audio_features)

u, s, v = np.linalg.svd(input_features, full_matrices=False)
v = v.T



pickle.dump(u, open('./models/saved_models/cfa_video.pickle','wb'))
pickle.dump(v, open('./models/saved_models/cfa_audio.pickle','wb'))
