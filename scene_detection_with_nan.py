import pickle
import numpy as np

nan_list = ['0Z_jjEHY3n4.pickle', 'K6NaXMsdMdw.pickle']

for name in nan_list:
    path = './nan_pickle/' + name
    features = np.array(pickle.load(open(path,'rb')))
    result = np.mean(features,axis=0)
    output_path = './' + name.split('.')[0] + '.pickle'
    pickle.dump(result,open(output_path,'wb'))
    print (len(result))
