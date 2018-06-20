import ffmpy
import os
from tqdm import tqdm

filename = list()
for root,dirs,files in os.walk("./youtube"):
    for f in files:
        filename.append(f.split(".")[0])

for root, dirs, files in os.walk("./youtube_mp3"):
    for f in files:
        name = f.split(".")[0]
        if name in filename:
            filename.remove(name)

for name in tqdm(filename):
    file_dir = "./youtube/" + name + ".mp4"
    out_dir = "./youtube_mp3/" + name + ".mp3"
    ff = ffmpy.FFmpeg(
        inputs = { file_dir: None},
        outputs = { out_dir: None} )
    ff.run()
