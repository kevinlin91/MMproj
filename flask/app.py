from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import subprocess
import pickle
import sys
import evaluation
import os
from moviepy.editor import *
import shutil
import glob
import keras.backend as K
from util import get_feature
import wgan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
import gc


def load_audio_features(path='D:/2018_MM_Project/audio_features_test/'):
   audio_feature_path = path
   audio_features = list()
   for root, dirs, files in os.walk(audio_feature_path):
      for f in files:
         new_path = audio_feature_path + f
         feature = pickle.load(open(new_path,'rb'))
         audio_features.append(feature)
   return audio_features

def scene_detection_parse(name):
    file_list = [name.split('.')[0]]
    for file_name in file_list:
        scene_path = './scene_detection/' + file_name + '.csv'
        scene_status = True
        parse_data = list()
        with open(scene_path,'r') as f:
            for i in range(3):
                f.readline()
            data = f.readlines()
            if len(data) < 2:
                scene_status=False
            if scene_status:
                parse_data = [ int(x.strip().split(',')[1]) for x in data if x!='\n']
            else:
                parse_data = [-1]
        pickle.dump(parse_data, open('./scene_detection_pickle/%s.pickle' % (name.split('.')[0]),'wb'))
        
def getLength(filename):

  result = subprocess.Popen(["ffprobe", filename],

    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)

  return [x for x in result.stdout.readlines() if "Duration" in x]

app = Flask(__name__)

@app.route('/upload')
def upload():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':
      f = request.files['file']
      name = f.filename
      f.save('./user_video/'+secure_filename(f.filename))
      with open('./file_name.txt','w') as f:
         f.write(name)
      return render_template('uploader.html')
   
#@app.route('/analysis')
#def loading_audio_feature():
#   return render_template('loading_audio_feature.html')
#@app.route('/ajax/loading_audio_feature')
#def ajax_loading_audio_feature():
#   audio_features = load_audio_features()
#   return redirect(url_for('extract_video_feature'))
   

@app.route('/analysis')
def extract_video_feature():
   return render_template('extract_video_feature.html')
@app.route('/ajax/extract_video_feature')
def ajax_extract_video_featrue():
   name = ''
   with open('./file_name.txt','r') as f:
      name = f.read()
   print (name)
   shell = 'scenedetect --input ./user_video/%s -d content -t 15 --csv-output ./scene_detection/%s.csv' % (name, name.split('.')[0])
   os.system(shell)
   scene_detection = scene_detection_parse(name)
   exec(compile(open('./video_feature_extraction.py', "rb").read(), './video_feature_extraction.py', 'exec'))
   K.clear_session()
   return redirect(url_for('bgm_recommendation'))

@app.route('/bgm_recommendation')
def bgm_recommendation():
   return render_template('bgm_recommendation.html')
@app.route('/ajax/bgm_recommendation')
def ajax_bgm_recommendation():
   #exec(compile(open('./evaluation.py', "rb").read(), './evaluation.py', 'exec'))
   #subprocess.call("evaluation.py", shell=True)
   EVA = evaluation.evaluation()
   wgan_result = EVA.gan_evaluation()
   
   with open('./recommend_result.txt','w') as f:
        f.write(wgan_result)
   return redirect(url_for('mix_video_audio'))
   
@app.route('/mix_video_audio')
def mix_video_audio():
   return render_template('mix_video_audio.html')
@app.route('/ajax/mix_video_audio')
def ajax_mix_video_auio():
   video_name = ''
   with open('./file_name.txt','r') as f:
      video_name = f.read()
   audio_name = ''
   with open('./recommend_result.txt','r') as f:
      audio_name = f.read()
   audio_name += '.mp3'
   
   files = glob.glob('./static/*')
   for f in files:
      if f.split('.')[1] == 'mp4':
         os.remove(f)
      
   video = VideoFileClip('./user_video/%s' % video_name)
   music = AudioFileClip("../youtube_mp3/%s" % audio_name)
   videoclip2 = video.set_audio(music)
   videoclip2.write_videofile("result.mp4")
   video.close()
   music.close()
   videoclip2.close()
   shutil.move("./result.mp4", "./static/result.mp4")
   return render_template('analysis_finish.html')
   
@app.route('/result_mv')
def result_mv():
   gc.collect()
   return render_template('result_mv.html')



if __name__ == '__main__':
   app.run(debug = True, host='0.0.0.0')
