'''
Description: Preprocess mp4 file, and output preprocessed frames and audio files
Output Format: 
            - Video Frames (.npy): central crop with dim=224 and only take the first {sampled_frames} (e.g. 64)
            - Audio (.npy): 

'''
import numpy as np
import scipy
import scipy.io.wavfile
from PIL import Image
import tempfile
import copy
import glob as _glob
import os
from params import *
from tqdm import tqdm
import torchaudio
import librosa


'''
   ========================
      Helper Functions
   ========================
'''
def glob(*args):
   return _glob.glob(os.path.join(*args))

def sortglob(*args):
   return sorted(glob(*args))

def centered(half, dim):
   s = half - dim/2.
   return s, s + dim

def sys_check_silent(*args):
  cmd = ' '.join(args)
  if 0 != os.system(cmd):
    fail('Command failed: %s' % cmd)
  return 0

def normalize_rms_np(samples, desired_rms = 0.1, eps = 1e-4):
   if len(samples.shape) == 2:
      rmse = librosa.feature.rms(samples.transpose((1,0)), frame_length=stft_frame_length, hop_length=stft_frame_step, center=True)[0]
   else:
      rmse = librosa.feature.rms(samples, frame_length=stft_frame_length, hop_length=stft_frame_step, center=True)[0]

   rmse = np.maximum(eps, rmse)
   length = samples.shape[0]
   num_complete = int(length/stft_frame_step)
   complete_samp = num_complete*stft_frame_step

   if len(samples.shape) == 2:
      major = desired_rms*samples[:complete_samp].reshape(num_complete, -1, 2) / np.expand_dims(np.expand_dims(rmse[:num_complete], axis=1),axis=2)
      remain = desired_rms*samples[complete_samp:] / rmse[num_complete]
      samples = np.concatenate((major.reshape(num_complete*stft_frame_step, 2), remain), axis=0)
   else:
      major = desired_rms*samples[:complete_samp].reshape(num_complete, -1) / np.expand_dims(rmse[:num_complete], axis=1)
      remain = desired_rms*samples[complete_samp:] / rmse[num_complete]
      samples = np.concatenate((major.reshape(num_complete*stft_frame_step, ), remain), axis=0)

   return samples

pjoin = os.path.join

def luminance_rgb(im): return rgb_from_gray(luminance(im))

def luminance(im):
   if len(im.shape) == 2:
      return im
   else:
      # see http://www.mathworks.com/help/toolbox/images/ref/rgb2gray.html
      return np.uint8(np.round(0.2989 * im[:,:,0] + 0.587 * im[:,:,1] + 0.114 * im[:,:,2]))

def rgb_from_gray(img, copy = True, remove_alpha = True):
  if img.ndim == 3 and img.shape[2] == 3:
    return img.copy() if copy else img
  elif img.ndim == 3 and img.shape[2] == 4:
    return (img.copy() if copy else img)[..., :3]
  elif img.ndim == 3 and img.shape[2] == 1:
    return np.tile(img, (1,1,3))
  elif img.ndim == 2:
    return np.tile(img[:,:,np.newaxis], (1,1,3))
  else:
    raise RuntimeError('Cannot convert to rgb. Shape: ' + str(img.shape))

def from_pil(pil):
   return np.array(pil)

def img_load(path, gray=False):
   im = from_pil(Image.open(path))
   if gray:
      return luminance(im)
   elif not gray and np.ndim(im) == 2:
      return rgb_from_gray(im)
   else:
      return im

def atleast_2d_col(x):
   x = np.asarray(x)
   if np.ndim(x) == 0:
      return x[np.newaxis, np.newaxis]
   if np.ndim(x) == 1:
      return x[:, np.newaxis]
   else:
      return x

def load_sound(wav_fname):
   rate, samples = scipy.io.wavfile.read(wav_fname)
   times = (1./rate) * np.arange(len(samples))
   return Sound(times, rate, samples)

class VidFrames:
   def __init__(self, vid_path, sound=True, start_time=None, 
         end_time=None, dims=None, sr=21000, fps=None):

      self.vid_path = vid_path
      self.sound = sound
      self.start_time = start_time
      self.end_time = end_time
      self.dims = dims
      self.sr = sr
      self.fps = fps

   def __enter__(self):
      if not os.path.exists(self.vid_path):
         raise RuntimeError('Video does not exist:' + self.vid_path)
      self.path = tempfile.mkdtemp()
      start_str = '-ss %f' % self.start_time if self.start_time is not None else ''
      dur_str = '-t %f' % (self.end_time - max(0, self.start_time)) if self.end_time is not None else ''
      dim_str = "-vf 'scale=%d:%d'" % self.dims if self.dims is not None else ''
      fps_str = '-r %f' % self.fps if self.fps is not None else ''
      sys_check_silent('ffmpeg -loglevel fatal %s -i "%s" %s %s %s "%s/%%07d.png"' % 
                        (start_str, self.vid_path, dim_str, dur_str, fps_str, self.path))
      if self.sound:
         sound_file = pjoin(self.path, 'sound.wav')
         '''
         sys_check_silent('ffmpeg -loglevel fatal %s -i "%s" %s -ac 2 -ar %d "%s"' % 
                           (start_str, self.vid_path, dur_str, self.sr, sound_file))
         '''
         sys_check_silent('ffmpeg -loglevel fatal -i "%s" -ac 2 -ar %d "%s"' % 
                           (self.vid_path, self.sr, sound_file))
      else:
         sound_file = None

      return sortglob(pjoin(self.path, '*.png')), sound_file 

   def __exit__(self, type, value, tb):
      import shutil
      shutil.rmtree(self.path)

class Sound:
   def __init__(self, times, rate, samples = None):

      if samples is None:
         samples = times
         times = None
      
      if samples.dtype == np.float32:
         samples = samples.astype('float64')

      self.rate = rate
      self.samples = atleast_2d_col(samples)
      self.length = samples.shape[0]

      if times is None:
         self.times = np.arange(len(self.samples)) / float(self.rate)
      else:
         self.times = times

   def copy(self):
      return copy.deepcopy(self)

   def parts(self):
      return (self.times, self.rate, self.samples)

   def __getslice__(self, *args):
      return Sound(self.times.__getslice__(*args), self.rate,
            self.samples.__getslice__(*args))

   def duration(self):
      return self.samples.shape[0] / float(self.rate)

   def normalized(self, check = True):
      if self.samples.dtype == np.double:
         assert (not check) or np.max(np.abs(self.samples)) <= 4.
         x = copy.deepcopy(self)
         x.samples = np.clip(x.samples, -1., 1.)
         return x
      else:
         s = copy.deepcopy(self)
         s.samples = np.array(s.samples, 'double') / np.iinfo(s.samples.dtype).max
         s.samples[s.samples < -1] = -1
         s.samples[s.samples > 1] = 1
         return s

   def unnormalized(self, dtype_name = 'int32'):
      s = self.normalized()
      inf = np.iinfo(np.dtype(dtype_name))
      samples = np.clip(s.samples, -1., 1.)
      samples = inf.max * samples
      samples = np.array(np.clip(samples, inf.min, inf.max), dtype_name)
      s.samples = samples
      return s

   def sample_from_time(self, t, bound = False):
      if bound:
         return min(max(0, int(np.round(t * self.rate))), self.samples.shape[0]-1)
      else:
         return int(np.round(t * self.rate))

   def shift_zero(self):
      s = copy.deepcopy(self)
      s.times -= s.times[0]
      return s

   def select_channel(self, c):
      s = copy.deepcopy(self)
      s.samples = s.samples[:, c]
      return s

   def left_pad_silence(self, n):
      if n == 0:
         return self.shift_zero()
      else:
         if np.ndim(self.samples) == 1:
            samples = np.concatenate([[0] * n, self.samples])
         else:
            samples = np.vstack([np.zeros((n, self.samples.shape[1]), self.samples.dtype), self.samples])
      return Sound(None, self.rate, samples)

   def right_pad_silence(self, n):
      if n == 0:
         return self.shift_zero()
      else:
         if np.ndim(self.samples) == 1:
            samples = np.concatenate([self.samples, [0] * n])
         else:
            samples = np.vstack([self.samples, np.zeros((n, self.samples.shape[1]), self.samples.dtype)])
      return Sound(None, self.rate, samples)

   def pad_slice(self, s1, s2):
      assert s1 < self.samples.shape[0] and s2 >= 0
      s = self[max(0, s1) : min(s2, self.samples.shape[0])]
      s = s.left_pad_silence(max(0, -s1))
      s = s.right_pad_silence(max(0, s2 - self.samples.shape[0]))
      return s


if __name__ == '__main__':

   raw_path = '../data/voxceleb/valid/'
   save_path = '../data/voxceleb/valid/'
   files = glob(raw_path, '*.mp4')
   print(files)
   print(f'Total files: {len(files)}')
   d = 224

   
   for i,f in enumerate(tqdm(files)):
      with VidFrames(vid_path=f, sound = True,
                     start_time = 0., end_time = vid_dur + 2./30 , dims=(256,256), fps = 29.97) \
                     as (im_files, snd_file):
         filename = f.split('/')[-1].split('.')[0]
         ims = np.array(list(map(img_load, im_files)))
         ## Central Crop and only take the sampled frames
         y = x = int(ims.shape[1]/2 - d/2)
         ims = ims[:, y : y + d, x : x + d]
         assert (ims.shape[0] >= sampled_frames), f"Video samples {ims.shape[0]} is smaller than the {sampled_frames} vid_dur = {vid_dur}!"
         ims = ims[:sampled_frames]
         np.save(save_path+'frames/'+filename+'.npy', ims)

         snd = load_sound(snd_file).normalized()
         snd_copy = snd.samples.copy()
         assert (snd_copy.shape[0] >= num_samples), f"Audio samples {snd_copy.shape[0]} is smaller than the {num_samples} vid_dur = {vid_dur}!"
         np.save(save_path+'audios/'+filename+'.npy', snd_copy[:num_samples])
   print('finish')

