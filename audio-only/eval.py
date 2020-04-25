import numpy as np
import librosa
from mir_eval.separation import bss_eval_sources_framewise
from pypesq import pesq
from tqdm import tqdm
from pypesq import pesq
from params import *
from preprocess import normalize_rms_np


test_path = './result/unet-set123/'

scores = {'sdr':[], 'sir': [], 'sar':[], 'pesq': []}

for idx in tqdm(range(2351, 2601)):
   gt = np.load(test_path + f'gt_{idx}.npy')
   if np.sum(gt) == 0:
      print('Silent File idx: ', idx)
      continue

   fg = np.load(test_path + f'fg_{idx}.npy')
   bg = np.load(test_path + f'bg_{idx}.npy')

   #fg = fg / np.max(np.abs(fg))
   #bg = fg / np.max(np.abs(bg))
   #gt = gt / np.max(np.abs(gt))

   fg = normalize_rms_np(np.clip(fg, -1.0, 1.0)).reshape(1,-1)
   bg = normalize_rms_np(np.clip(bg, -1.0, 1.0)).reshape(1,-1)
   np.save('fg.npy', fg)
   np.save('bg.npy', bg)
   aaa
   gt = normalize_rms_np(gt).reshape(1,-1)

   sdr1,sir1,sar1,_ = bss_eval_sources_framewise(gt, fg)
   sdr2,sir2,sar2,_ = bss_eval_sources_framewise(gt, bg)
   sdr = np.maximum(sdr1, sdr2)
   sir = np.maximum(sir1, sir2)
   sar = np.maximum(sar1, sar2)

   #print(f'File: {idx}, sdr={sdr}, sir={sir}, sar={sar}')
   scores['sdr'].append(sdr)
   scores['sir'].append(sir)
   scores['sar'].append(sar)

   gt = librosa.core.resample(gt, 21000, 16000)
   fg = librosa.core.resample(fg, 21000, 16000)
   bg = librosa.core.resample(bg, 21000, 16000)

   esq = pesq(gt[0], fg[0], 16000)
   esq_ = pesq(gt[0], bg[0], 16000)
   
   if np.isnan(esq) and np.isnan(esq_):
      print(f'Nan pesq in file {idx}')
      print(np.sum(gt))
      print(np.sum(fg))
      print(np.sum(bg))
   elif np.isnan(esq):
      scores['pesq'].append(esq_)
   elif np.isnan(esq_):
      scores['pesq'].append(esq)
   else:
      scores['pesq'].append(np.maximum(esq, esq_))


print(f"\nPESQ: {np.mean(scores['pesq'])}, SDR: {np.mean(scores['sdr'])}, SIR: {np.mean(scores['sir'])}, SAR: {np.mean(scores['sar'])}\n")
