from pytube import YouTube
from tqdm import tqdm

with open('./Youtube_ID.txt','r') as f:
    data = f.readlines()
data = [ x.rstrip() for x in data]
for youtube_link in tqdm(data[5501:5504]):
    try:
        yt = YouTube(youtube_link)
        _id = youtube_link.split('=')[1]
        video = yt.streams.filter(file_extension='mp4').first()
        video.download('./youtube',filename=_id)
    except:
        print (youtube_link)
        with open('./error_link.txt','a') as f:
            s = youtube_link + " mp4 do not exist or link was revomed\n"
            f.write(s)
        

    
