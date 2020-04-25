from tqdm import tqdm
import numpy as np
import datetime
import json
import os

def read_json(filepath):
   with open(filepath, 'r', newline='\n', encoding='utf-8') as f:
      d = json.load(f)

   return d

def video_download(loc, url_list):
    """
    Args:
        # Only download the video from the link
        # loc        | the location for downloaded file
        # v_name     | the name for the video file
        # cat        | the catalog with audio link and time
    """
    for i, url in enumerate(tqdm(url_list)):
        command = 'cd %s;' % loc
        f_name = str(i)
        link = "https://www.youtube.com/watch?v="+url
        start_time = 10
        end_time = start_time + 5.0
        start_time = datetime.timedelta(seconds=start_time)
        end_time = datetime.timedelta(seconds=end_time)
        command += 'ffmpeg -i $(youtube-dl -f ”mp4“ --get-url ' + link + ') ' + '-c:v h264 -c:a copy -ss %s -to %s %s.mp4' \
                % (start_time, end_time, f_name)
        os.system(command)

if __name__ == "__main__":

   d = read_json('training_list.json')
   print("Total number of ids:", len(d.keys()))

   id_num = 150
   id_list = np.random.choice(list(d.keys()), id_num)

   print("=========================================\n")
   print(f"Randomly select {id_num} Person's id")
   print("\n=========================================")

   url_list = [np.random.choice(d[yid]['url_list'], 1)[0] for yid in id_list]

   print("=========================================\n")
   print("Randomly select YT URL based on Person id")
   print("\n=========================================")

   print(url_list)
   video_download('./train/', url_list)

