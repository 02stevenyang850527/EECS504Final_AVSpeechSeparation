import librosa
import argparse
import numpy as np
from tqdm import tqdm
import scipy.io.wavfile as wavfile
import glob as _glob
import os
from params import *

assert (duration_mult == 1.), "Because we only have 3 sec waveform .."

def glob(*args):
   return _glob.glob(os.path.join(*args))

if __name__ == '__main__':


   parser = argparse.ArgumentParser()
   parser.add_argument('-s', '--size', help='size of noise set', type=int, default=0)
   parser.add_argument('-r', '--range', help='size of noise set', type=int, default=0)
   args = parser.parse_args()

   
   #root_path = '/home/ubuntu/eecs504/avspeech/data/audio_data/norm_audio_train/'
   #fileslist = list(range(1, args.range+1))

   root_path = '/home/ubuntu/eecs504/ESC-50-master/audio/'
   fileslist = glob(root_path, '*.wav')
   print('Total noise set:', len(fileslist))
   output_path = f'../data/noise_set/noise_{args.size}/'

   cnt = 1
   for idx in tqdm(fileslist):
      try:
         #rate, snd = wavfile.read(root_path + f'trim_audio_train{idx}.wav')
         rate, snd = wavfile.read(idx)
         if isinstance(snd[0], np.int16):
            snd = snd / 32767.0
         cnt += 1
      except FileNotFoundError:
         print(f'trim_audio_train{idx}.wav is missing')
         continue

      new_snd = librosa.core.resample(snd, rate, int(samp_sr))
      assert (new_snd.shape[0] >= num_samples), f"Audio samples {new_snd.shape[0]} is smaller than the {num_samples} vid_dur = {vid_dur}!"
      np.save(output_path + f'{cnt}.npy', new_snd[:num_samples])
