import os
import pickle
import numpy as np

def scene_detection_parse():
    file_list = list()
    for root, dirs, files in os.walk('./final_scene_detection/', 'r'):
        file_list = files
    for file_name in file_list:
        scene_path = './final_scene_detection/' + file_name
        scene_status = True
        parse_data = list()
        with open(scene_path,'r') as f:
            for i in range(3):
                f.readline()
            data = f.readlines()
            if len(data) < 1:
                scene_status=False
            if scene_status:
                parse_data = [ int(x.strip().split(',')[1]) for x in data if x!='\n']
            else:
                parse_data = [-1]
                
        pickle_path = './final_video_features/' + file_name.split('.')[0] + '.pickle'
        feature_data = np.array(pickle.load(open(pickle_path,'rb')))
        result = list()
        if scene_status:
            parse_index = [ i for x in parse_data if x < (len(feature_data)-1) for i in (x-1,x+1) ]
            tmp_feature_data = feature_data[parse_index]
            tmp = np.mean(tmp_feature_data,axis=0)
            result = tmp
        else:
            tmp = np.mean(feature_data,axis=0)
            result = tmp
        output_path = './final_video_features_with_scene/' + file_name.split('.')[0] + '.pickle'
        print (output_path)
        pickle.dump(result,open(output_path,'wb'))
        
        
    

if __name__=='__main__':
    scene_detection_parse()
