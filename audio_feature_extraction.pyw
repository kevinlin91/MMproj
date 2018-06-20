from sys import argv  # allows user to specify input and output directories
import os  # help with file handling
import librosa_feature
import warnings

warnings.filterwarnings('ignore')

SR = 12000
N_FFT = 512
HOP_LEN = 256
DURA = 29.12

ext = '.mp3'
indir = './music_library'
outdir = './music_library_features'

files = [] 
filelist = [f for f in os.listdir(indir) if f.endswith(ext)]
for path in filelist:
    basename = os.path.basename(path)
    filename = os.path.splitext(basename)[0]
    files.append(filename)


for filename in files:
    librosa_feature.feature_extraction(indir, filename, ext, outdir, SR, N_FFT, HOP_LEN, DURA)


