import os
import copy
import torch
import argparse
import torchaudio
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from tqdm import trange
from params import *
from preprocess import *
from numba import njit, prange

print("PyTorch Version: ",torch.__version__)
print("Torchaudio Version: ",torchaudio.__version__)
print("Num Samples in waveform", num_samples)
print('Spectrogram samples:', spec_len)
print('Spectrogram Window length:', stft_frame_length)
print('Spectrogram Window step:', stft_frame_step)
print('Spectrogram FFT taps:', stft_num_fft)


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
   print("Using the GPU!")
else:
   print("WARNING: Could not find GPU! Using CPU only")


def count_params(model):
   '''
   Counts the number of trainable parameters in PyTorch.

   Args:
       model: PyTorch model.

   Returns:
       num_params: int, number of trainable parameters.
   '''

   num_params = sum([item.numel() for item in model.parameters() if item.requires_grad])

   return num_params

'''
   ======================
      HELPER Functions
   ======================
'''
def norm_range(x, min_val, max_val):
   return 2.*(x - min_val)/float(max_val - min_val) - 1.


def unnorm_range(y, min_val, max_val):
   return 0.5*float(max_val - min_val) * (y + 1) + min_val


def angle(z):
   """
   Returns the elementwise arctan of z, choosing the quadrant correctly.
   Quadrant I: arctan(y/x)
   Qaudrant II: \pi + arctan(y/x) (phase of x<0, y=0 is \pi)
   Quadrant III: -\pi + arctan(y/x)
   Quadrant IV: arctan(y/x)
   Inputs:
      z: tf.complex64 or tf.complex128 tensor
   Retunrs:
      Angle of z
   """
   #return torch.atan2(torch.imag(z), torch.real(z))
   return torchaudio.functional.angle(z)


def db_from_amp(x):
  #return 20. * torch.log10(torch.max(x, torch.from_numpy(np.array([1e-5])))) # Double
  return 20. * torch.log10(torch.max(x, torch.Tensor([1e-5])))


def amp_from_db(x):
  return torch.pow(10., x / 20.)


def img_load_scale(path, dim=256, gray=False):
   im = Image.open(path)
   im_resize = from_pil(im.resize((dim,dim)))
   if gray:
      return luminance(im_resize)
   elif not gray and np.ndim(im) == 2:
      return rgb_from_gray(im_resize)
   else:
      return im_resize

def make_complex(mag, phase):
   mag = mag.unsqueeze_(3)
   phase = phase.unsqueeze_(3)
   z = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], axis=3)
   return z


def istft(mag, phase, window):
   mag = amp_from_db(mag)
   complex_spec = make_complex(mag, phase)
   complex_spec = complex_spec.permute(0,2,1,3)
   waveform = torchaudio.functional.istft(complex_spec, n_fft=stft_num_fft, hop_length=stft_frame_step, 
                                         win_length=stft_frame_length, window=window)
   return waveform


def stft(x, window, mono=True):
   # Output shape:
   # (batch, channel, freq, time, complex=2), if Mono => (batch, freq, time, complex=2)
   if mono:
      x = x.mean(axis=0)
      complex_spec = torchaudio.functional.spectrogram(x, pad=0, window=window, 
                                                       n_fft=stft_num_fft, hop_length=stft_frame_step,
                                                       win_length=stft_frame_length, power=None, normalized=False)[:,:spec_len,:]
   else:
      complex_spec = torchaudio.functional.spectrogram(x, pad=0, window=window, 
                                                       n_fft=stft_num_fft, hop_length=stft_frame_step,
                                                       win_length=stft_frame_length, power=None, normalized=False)[:,:,:spec_len,:]
   return complex_spec


def generate_wav(mag, phase, window, p2d, pad_phase):
   pad_val = -100.
   mag = F.pad(mag, p2d, mode='constant', value=pad_val)
   phase = torch.cat((phase, pad_phase), axis=2)
   wav = istft(mag, phase, window)
   return wav


def predict_vidsep(mp4_path, model, noise_path=None):
   d = 224
   res = {}
   with VidFrames(vid_path=mp4_path, sound = True,
                  start_time = 0., end_time = clip_dur + 2./30 ,fps = 29.97) as (im_files, snd_file):
     ims = np.array(list(map(img_load_scale, im_files)))
     ims_resize = np.array(list(map(img_load, im_files))).astype(np.float32)

     y = x = int(ims_resize.shape[1]/2 - d/2)
     ims_resize = ims_resize[:, y : y + d, x : x + d]
     ims_resize = ims_resize[:sampled_frames]

     snd_orig = load_sound(snd_file)
     snd = snd_orig.copy()
     snd = snd.normalized()
     snd_copy = snd.samples.copy()
     assert (snd_copy.shape[0] >= num_samples), f"Video samples {snd_copy.shape[0]} is smaller than the {num_samples}!"
     
   model.eval() # Turn on the Evaluation Mode
   #input_rms = np.sqrt(0.1**2 + 0.1**2)
   #samples = normalize_rms_np(snd_copy[:num_samples], input_rms)
   samples = snd_copy[:num_samples]
   res['source'] = samples.copy()
   samples = normalize_rms_np(samples)

   if noise_path is not None:
      noise_set = np.array(glob(noise_path, '*.npy'))
      noise_file = np.random.choice(noise_set, 1)[0]
      noise = np.load(noise_file)
      res['noise'] = noise.copy()
      noise = normalize_rms_np(noise)
      samples = np.clip(samples + np.expand_dims(noise,axis=1), -1., 1.)
      res['mix_sound'] = samples
   '''
   else:
      print('Manually mix the noise')
      noise = np.load('../data/obamanet/valid/audios/2321.npy')
      res['noise'] = noise.copy().mean(axis=-1)
      noise = normalize_rms_np(noise)
      samples = np.clip(samples + noise, -1., 1.)
      res['mix_sound'] = samples
   '''

   ims_resize = np.expand_dims(ims_resize, axis=0)

   samples = np.expand_dims(samples, axis=0)

   vids = torch.from_numpy(ims_resize)
   vids = vids.permute(0,4,1,2,3)
   snds = torch.from_numpy(samples.astype(np.float32))
   snds = snds.permute(0,2,1)

   window = torch.hamming_window(stft_frame_length)
   complex_norm = torchaudio.transforms.ComplexNorm(1)
   complex_spec = stft(snds[0], window, mono=True)
   complex_spec = complex_spec.unsqueeze_(0)
   specgram = complex_spec.permute(0,2,1,3)

   spec_mag = db_from_amp(complex_norm(specgram))
   spec_phase = angle(specgram)
   
   # Transfer to device (GPU)
   vids = vids.to(device)
   snds = snds.to(device)
   spec_mag = spec_mag.to(device)
   spec_phase = spec_phase.to(device)

   # Compute the output
   pred_fg_mag, pred_fg_phase, pred_bg_mag, pred_bg_phase = model(snds, vids, spec_mag, spec_phase)

   # Pad to the original size
   pad_val = -100.
   p2d = (0, spec_mag.shape[-1]-freq_len)
   fg_wav = generate_wav(pred_fg_mag, pred_fg_phase, window.to(device), p2d, spec_phase[...,-1:])
   bg_wav = generate_wav(pred_bg_mag, pred_bg_phase, window.to(device), p2d, spec_phase[...,-1:])

   '''
   res['pred_fg'] = fg_wav.cpu().detach().numpy()
   res['pred_bg'] = bg_wav.cpu().detach().numpy()

   res['pred_fg'] = res['pred_fg'] / np.max(np.abs(res['pred_fg']))
   res['pred_bg'] = res['pred_bg'] / np.max(np.abs(res['pred_bg']))
   '''
   res['pred_fg'] = normalize_rms_np(np.clip(fg_wav.cpu().detach().numpy(), -1., 1.)[0])
   res['pred_bg'] = normalize_rms_np(np.clip(bg_wav.cpu().detach().numpy(), -1., 1.)[0])

   return res


# For UNet initialization
def normal_init(m, mean=0.0, std=0.02):
   if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
      nn.init.normal_(m.weight, mean, std)
      if m.bias is not None:
         nn.init.constant_(m.bias, 0)


# For Shift Net initialization
# In original implementaion, it uses tf.VarianceScaling
def kaiming_init(m):
   if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight)
      if m.bias is not None:
         nn.init.constant_(m.bias, 0)

   
'''
=============================
'''

class Block3(nn.Module):
   def __init__(self, dim_in, dim_out, stride=1, use_bn=True, use_pool=False):
   # Residual-like block
      super().__init__()
      self.block3 = nn.Sequential(
                     nn.Conv3d(dim_in, dim_out, (3,3,3), stride=stride, padding=1),
                     nn.BatchNorm3d(dim_out),
                     nn.ReLU(),
                     nn.Conv3d(dim_out, dim_out, (3,3,3), padding=1)
                  )
      if stride == 1:
         self.short = nn.Sequential(nn.Identity())
      else:
         if use_pool:
            self.short = nn.Sequential(nn.MaxPool3d((1,1,1), stride=stride))
         else:
            self.short = nn.Sequential(nn.Conv3d(dim_in, dim_out, (1,1,1), stride=stride),
                                       nn.BatchNorm3d(dim_out)
                                    )

      if use_bn:
         self.norm_act = nn.Sequential(nn.BatchNorm3d(dim_out), nn.ReLU())
      else:
         self.norm_act = nn.Sequential(nn.ReLU())

   def forward(self, x):
      return self.norm_act(self.block3(x) + self.short(x))


class Block2(nn.Module):
   def __init__(self, dim_in, dim_out, stride=(4,1), use_pool=False):
   # Residual-like block
      super().__init__()
      self.block2 = nn.Sequential(
                     nn.Conv2d(dim_in, dim_out, (15,1), stride=stride, padding=(7,0)),
                     nn.BatchNorm2d(dim_out),
                     nn.ReLU(),
                     nn.Conv2d(dim_out, dim_out, (15,1), padding=(7,0))
                  )
      if stride == 1:
         self.short = nn.Sequential(nn.Identity())
      else:
         if use_pool:
            self.short = nn.Sequential(nn.MaxPool2d((1,1), stride=stride))
         else:
            self.short = nn.Sequential(nn.Conv2d(dim_in, dim_out, (1,1), stride=stride),
                                       nn.BatchNorm2d(dim_out)
                                    )

      self.norm_act = nn.Sequential(nn.BatchNorm2d(dim_out), nn.ReLU())

   def forward(self, x):
      return self.norm_act(self.block2(x) + self.short(x))


class ImNet(nn.Module):
   def __init__(self):
      super().__init__()
      self.net = nn.Sequential(
               nn.Conv3d(3, 64, (5,7,7), stride=2, padding=(2,3,3)),
               nn.BatchNorm3d(64),
               nn.ReLU(),
               nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))
               )
      self.block3_1 = Block3(64, 64, 1)
      self.block3_2 = Block3(64, 64, 2)

   def forward(self, x):
   # Image should be preprocessed: Image = -1. + (2./255) * x
      im_pool = self.net(x)
      im_block1 = self.block3_1(im_pool)
      im_block2 = self.block3_2(im_block1) 

      return im_block2


class SfNet(nn.Module):
   def __init__(self):
      super().__init__()
      self.net = nn.Sequential(
               nn.Conv2d(2, 64, (65, 1), stride=4, padding=(32,0)),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.MaxPool2d((4, 1), stride=(4,1))
            )
      self.block2_1 = Block2(64, 128, use_pool=False)
      self.block2_2 = Block2(128, 128, use_pool=True)
      self.block2_3 = Block2(128, 256, use_pool=False)

   def forward(self, x):
   # Sound features should be normalized
      sf_pool = self.net(x)
      sf_block1 = self.block2_1(sf_pool)
      sf_block2 = self.block2_2(sf_block1)
      sf_block3 = self.block2_3(sf_block2)

      return sf_block3

# Used to Merge ImNet and SfNet
class MergeLayer(nn.Module):
   def __init__(self):
      super().__init__()
      self.match = nn.Sequential(
                     nn.Conv2d(256, 128, (3,1), padding=(1,0)),
                     nn.BatchNorm2d(128),
                     nn.ReLU()
                  )
      self.merge = nn.Sequential(
                     nn.Conv3d(192, 512, (1,1,1)),
                     nn.BatchNorm3d(512),
                     nn.ReLU(),
                     nn.Conv3d(512, 128, (1,1,1))
                  )
      self.norm_act = nn.Sequential(
                     nn.BatchNorm3d(128),
                     nn.ReLU()
                  )

   def forward(self, sf, im):
      s0 = sf.shape
      frac = float(s0[2]-1)/im.shape[2]
      outs = (int(s0[2]/frac), 1)
      sf = F.adaptive_max_pool2d(sf, output_size=outs)
      matched_sf = self.match(sf)
      matched_sf = matched_sf[:,:,:,0]
      expand_sf = matched_sf.unsqueeze_(3).unsqueeze_(4)
      tile_sf = expand_sf.repeat(1,1,1,28,28)
      concat = torch.cat((im, tile_sf), axis=1)
      short = torch.cat((concat[:,:64,:,:,:], concat[:,-64:,:,:,:]), axis=1)
      merge1 = self.merge(concat)

      return self.norm_act(merge1 + short)

'''
   Sample Input Shape:
   E.g.
	Image input shape: [1, 3, 248, 224, 224] -> Output shape of ImNet: [1, 64, 62, 28, 28] (N x C x D x H x W)
      Sound input shape: [1, 2, 173774] (N x C x L) -> Output shape of SfNet: [1, 256, 170, 1] (N x C x H x W)
'''

class ShiftNet(nn.Module):
   ''' 
   ====== Input Format =========
    image: -1 x 3 x sampled_frames x crop_im_dim, crop_im_dim (N x C x D x H x W)
    samples: -1 x 2 x num_samples (N x C x L)
   ==============================
   ''' 
   def __init__(self):
      super().__init__()
      self.imnet = ImNet() # Image subnetwork
      self.sfnet = SfNet() # Sound feature subnetwork
      self.merge = MergeLayer() # Merge Image and Sound feature
      self.block3_1 = Block3(128, 128, 1)
      self.block3_2 = Block3(128, 128, 1)
      self.block3_3 = Block3(128, 256, 2)
      self.block3_4 = Block3(256, 256, 1) 
      self.block3_5 = Block3(256, 512, (1,2,2))
      self.block3_6 = Block3(512, 512, 1) 
      

   def forward(self, audio, images):
      normalize_audio = torch.sign(audio)*(torch.log(1 + 255.*torch.abs(audio)) / np.log(1 + 255))
      normalize_audio = normalize_audio.unsqueeze_(3)
      normalize_image = -1. + (2./255) * images
      imf = self.imnet(normalize_image)
      sf =  self.sfnet(normalize_audio)
      mf = self.merge(sf, imf)
      net = self.block3_1(mf)
      scale1 = self.block3_2(net)
      scale2 = self.block3_3(scale1)
      scale2 = self.block3_4(scale2)
      scale2 = self.block3_5(scale2)
      last_conv = self.block3_6(scale2)

      # return scale0, scale1, scale2
      return imf, scale1, last_conv

class UnetConv(nn.Module):
   def __init__(self, dim_in, dim_out, stride=2):
      super().__init__()
      self.conv = nn.Sequential(nn.Conv2d(dim_in, dim_out, 3, stride=stride, padding=1),
                                nn.BatchNorm2d(dim_out)
                  )
      self.act  = nn.Sequential(nn.LeakyReLU(0.2))

   def forward(self, x):
      out = self.conv(x)
      return out, self.act(out)


class UnetDeConv(nn.Module):
   def __init__(self, dim_in, dim_out, stride=2, use_cat=True, use_bn=True):
      super().__init__()
      self.cat = use_cat
      op = 1 if stride==2 else [int(i/2) for i in stride]
      if use_bn:
         self.deconv = nn.Sequential(nn.ReLU(), 
                                     nn.ConvTranspose2d(dim_in, dim_out, 3, stride=stride, padding=1, output_padding=op),
                                     nn.BatchNorm2d(dim_out)
                                    )
      else:
         self.deconv = nn.Sequential(nn.ReLU(), 
                                     nn.ConvTranspose2d(dim_in, dim_out, 3, stride=stride, padding=1, output_padding=op),
                                     nn.BatchNorm2d(dim_out)
                                    )

   def forward(self, x, skip_layer):
      if self.cat:
         x = torch.cat((x, skip_layer), 1)
      out = self.deconv(x)
      return out


class MergeShiftNetLayer(nn.Module):
   def __init__(self, merge=True):
      super().__init__()
      self.merge = merge
      self.Id = nn.Identity()

   def forward(self, net, vid_net):
      net = self.Id(net)
      if self.merge:
         vid_net = torch.mean(vid_net, dim=(3,4), keepdim=True)
         vid_net = vid_net[:,:,:,0,:]
         s = net.shape
         if s[2] != vid_net.shape[2]:
            # which means sampling rate is not matched..
            vid_net = F.interpolate(vid_net, size=(s[2],1), mode='bilinear')
         net = torch.cat( (net, vid_net.repeat(1,1,1,s[3])), 1)
         return net, net
      else:
         return net, net


class VidSepNet(nn.Module):
   def __init__(self, Full=True):
      super().__init__()
      self.Full = Full
      if Full:
         self.shift_net = ShiftNet()
      self.conv1 = UnetConv(  2,  64, (1,2))
      self.conv2 = UnetConv( 64, 128, (1,2))
      self.conv3 = UnetConv(128, 256, 2)
      if Full:
         self.merge1 = MergeShiftNetLayer(Full)
         self.conv4 = UnetConv(320, 512, 2)
         self.merge2 = MergeShiftNetLayer(Full)
         self.conv5 = UnetConv(640, 512, 2)
         self.merge3 = MergeShiftNetLayer(Full)
         self.conv6 = UnetConv(1024, 512, 2)
      else:
         self.conv4 = UnetConv(256, 512, 2)
         self.conv5 = UnetConv(512, 512, 2)
         self.conv6 = UnetConv(512, 512, 2)         
      self.conv7 = UnetConv(512, 512, 2)
      self.conv8 = UnetConv(512, 512, 2)
      self.conv9 = UnetConv(512, 512, 2)

      self.deconv1 = UnetDeConv(512, 512, 2, use_cat=False)
      self.deconv2 = UnetDeConv(1024, 512, 2)
      self.deconv3 = UnetDeConv(1024, 512, 2)
      self.deconv4 = UnetDeConv(1024, 512, 2)
      if Full:
         self.deconv5 = UnetDeConv(1536, 512, 2)
         self.deconv6 = UnetDeConv(1152, 256, 2)
         self.deconv7 = UnetDeConv(576, 128, 2)
         self.deconv8 = UnetDeConv(256,  64, (1,2))
      else:
         self.deconv5 = UnetDeConv(1024, 512, 2)
         self.deconv6 = UnetDeConv(1024, 256, 2)
         self.deconv7 = UnetDeConv(512, 128, 2)
         self.deconv8 = UnetDeConv(256, 64, (1,2))

      self.out_fg = UnetDeConv(128, 2, (1, 2), use_bn=False)
      self.out_bg = UnetDeConv(128, 2, (1, 2), use_bn=False)

   def weight_init(self, mean, std):
      for m in self._modules:
         if m == 'shift_net':
            kaiming_init(self._modules[m])
         else:
            normal_init(self._modules[m], mean, std)


   def forward(self, audio, images, spec, phase):
      '''
         Input format after truncate: [1, 2, 512, 1024] (N x D x H x W)
      '''
      if self.Full:
         scale0, scale1, scale2 = self.shift_net(audio, images)
      spec  = norm_range(spec, spec_min, spec_max)
      phase = norm_range(phase, -np.pi, np.pi)

      Unet_in = torch.cat((spec.unsqueeze_(1), phase.unsqueeze_(1)), dim=1)
      net = Unet_in[:,:,:,:freq_len]

      out1, net = self.conv1(net)
      out2, net = self.conv2(net)
      out3, net = self.conv3(net)
      if self.Full:
         out3, net = self.merge1(net, scale0)
      out4, net = self.conv4(net)
      if self.Full:
         out4, net = self.merge2(net, scale1)
      out5, net = self.conv5(net)
      if self.Full:
         out5, net = self.merge3(net, scale2)
      out6, net = self.conv6(net)
      out7, net = self.conv7(net)
      out8, net = self.conv8(net)
      out9, net = self.conv9(net)

      net = self.deconv1(net, out9)
      net = self.deconv2(net, out8)
      net = self.deconv3(net, out7)
      net = self.deconv4(net, out6)
      net = self.deconv5(net, out5)
      net = self.deconv6(net, out4)
      net = self.deconv7(net, out3)
      net = self.deconv8(net, out2)

      fg = self.out_fg(net, out1)
      bg = self.out_bg(net, out1)

      fg_spec  = unnorm_range(torch.tanh(fg[:,0,:,:]), spec_min, spec_max)
      fg_phase = unnorm_range(torch.tanh(fg[:,1,:,:]), -np.pi, np.pi)
      bg_spec  = unnorm_range(torch.tanh(bg[:,0,:,:]), spec_min, spec_max)
      bg_phase = unnorm_range(torch.tanh(bg[:,1,:,:]), -np.pi, np.pi)

      return fg_spec, fg_phase, bg_spec, bg_phase


class MixSound(object):
   def __init__(self, av_noise=True):
      self.av_noise = av_noise # Use noise from av speech dataset or not
      self.window = torch.hamming_window(stft_frame_length)
      self.complex_norm = torchaudio.transforms.ComplexNorm(1) # get the magnitude of compelx number

   def __call__(self, sample):
      snds = sample['snd']
      n = int(snds.shape[0]/2)

      if self.av_noise:
         noise = sample['noise']
      else:
         noise = torch.cat((snds[n:], snds[:n]),axis=0)

      samples_mix = torch.clamp(snds + noise, -1., 1.)

      # (channel, freq, time, complex=2)
      spec_0   = stft(snds, self.window, mono=True)
      spec_1   = stft(noise, self.window, mono=True)
      spec_mix = stft(samples_mix, self.window, mono=True)

      spec_0 = spec_0.permute(1,0,2)
      spec_1 = spec_1.permute(1,0,2)
      spec_mix = spec_mix.permute(1,0,2)

      fg_mag = db_from_amp(self.complex_norm(spec_0))
      fg_phase = angle(spec_0)

      bg_mag = db_from_amp(self.complex_norm(spec_1))
      bg_phase = angle(spec_1)

      spec_mag = db_from_amp(self.complex_norm(spec_mix))
      spec_phase = angle(spec_mix)
   
      return {'mix_mag': spec_mag, 'mix_phase': spec_phase, 'samples_mix': samples_mix, 'fg_samples': snds, 'bg_samples': noise,
            'fg_mag' : fg_mag, 'fg_phase': fg_phase, 'bg_mag': bg_mag, 'bg_phase': bg_phase}

class NormalizeRms(object):
   def __init__(self, desired_rms=0.1, eps=1e-4):
      self.desired_rms = 0.1
      self.eps = eps

   def __call__(self, sample):
      for k, v in sample.items():
         # input format (time, channel(optional))
         v = normalize_rms_np(v, self.desired_rms, self.eps)
         sample[k] = v
      return sample


class SndToTensor(object):
   def __call__(self, sample):
      for k, v in sample.items():
         #v = normalize_rms_np(v).astype(np.float32)
         v = v.astype(np.float32)
         v = torch.from_numpy(v)
         if len(v.shape) < 2:
            v = v.unsqueeze_(1)
      
         v =  v.permute(1,0)
         sample[k] = v

      return sample

class VidToTensor(object):
   """Convert ndarrays in sample to Tensors."""
   def __call__(self, sample):
      vids = torch.from_numpy(sample)
      vids = vids.permute((3,0,1,2)) # DxHxWxC -> CxDxHxW
      return vids

class RandomHorizontalFlip(object):
   def __init__(self, p):
      self.p = p

   def __call__(self, sample):
      flips = np.random.choice([True, False], sample.shape[0], p=[self.p, 1.-self.p]) 
      return np.array([np.fliplr(img) if flip else img for img, flip in zip(sample, flips)])

class VidSepDataset(Dataset):
   def __init__(self, vid_path, snd_path, noise_path=None, vid_transform=None, snd_transform=None):
      self.vid_file = np.array(glob(vid_path, '*.npy'))
      self.snd_file = np.array(glob(snd_path, '*.npy'))
      if noise_path is not None:
         self.noise=True
         self.noise_file = np.array(glob(noise_path, '*.npy'))
      else:
         self.noise = False
      self.vid_transform = vid_transform
      self.snd_transform = snd_transform

      assert (len(self.vid_file) == len(self.snd_file)), f"# videos: {len(self.vid_file)}; # audio: {len(self.snd_file)} should be the SAME!"

   def __len__(self):
      return len(self.vid_file)

   def __getitem__(self, idx):
      if torch.is_tensor(idx):
         idx = idx.tolist()

      vid_path = self.vid_file[idx]
      snd_path = self.snd_file[idx]
      vid = np.load(vid_path)
      snd = np.load(snd_path)
      if self.noise:
         noise_path = np.random.choice(self.noise_file, 1)[0]
         noise = np.load(noise_path)

      if self.vid_transform:
         vid = self.vid_transform(vid)

      samples = {}
      samples['snd'] = snd
      if self.noise:
         samples['noise'] = noise

      if self.snd_transform:
         samples = self.snd_transform(samples)

      return vid, samples


def compute_loss(criterion, pred, gt, min_val=None, max_val=None, use_norm=True):
   if use_norm:
      pred = norm_range(pred, min_val, max_val)
      gt = norm_range(gt, min_val, max_val)
      return criterion(pred, gt)
   else:
      return criterion(pred, gt)


def sum_loss(criterion, pred_mag, pred_phase, mag, phase, use_norm=True, phase_weight=0.01):
   mag_loss = compute_loss(criterion, mag, pred_mag, spec_min, spec_max, use_norm)
   phase_loss = compute_loss(criterion, phase, pred_phase, -np.pi, np.pi, use_norm)
   loss = mag_loss + phase_weight * phase_loss
   return loss


def train(model, dataloader, optimizer, min_loss=np.inf, num_epochs=3, 
          save_dir=None, use_norm=True, is_training=True, PIT=True):

   hist_loss = {'fg_loss': np.zeros((num_epochs,)), 'bg_loss': np.zeros((num_epochs,)), 'total_loss': np.zeros((num_epochs))}
   if PIT:
      print("USING PIT LOSS FUNCTION")

   if not is_training and not use_norm:
      window = torch.hamming_window(stft_frame_length).to(device)
   L1_loss = nn.L1Loss().cuda()

   best_model_wts = copy.deepcopy(model.state_dict())

   phase_weight = 0.01
   if is_training:
      model.train()
   else:
      model.eval()

   pad_val = -100. # in dB 
   silent_list = [] # to detect the silent source
   file_cnt = 2351

   for epoch in range(num_epochs):
      fg_loss_hist = []
      bg_loss_hist = []
      total_loss_hist = []
      pbar = tqdm(dataloader)
      for batch_vids, ret in pbar:
         # Get a batch of data from dataset

         # Transfer them to device (GPU)
         batch_vids = batch_vids.to(device)
         samples_mix = ret['samples_mix'].to(device)
         mix_mag = ret['mix_mag'].to(device)
         mix_phase = ret['mix_phase'].to(device)
         fg_mag = ret['fg_mag'].to(device)
         bg_mag = ret['bg_mag'].to(device)
         fg_phase = ret['fg_phase'].to(device)
         bg_phase = ret['bg_phase'].to(device)

         # Predict the fg/bg spectrogam
         if is_training:
            optimizer.zero_grad()
         pred_fg_mag, pred_fg_phase, pred_bg_mag, pred_bg_phase = model(samples_mix, batch_vids, mix_mag, mix_phase)

         # Pad to freq length
         p2d = (0, fg_mag.shape[-1]-freq_len)
         pred_fg_mag = F.pad(pred_fg_mag, p2d, mode='constant', value=pad_val)
         pred_fg_phase = torch.cat((pred_fg_phase, fg_phase[...,-1:]), axis=2)
         pred_bg_mag = F.pad(pred_bg_mag, p2d, mode='constant', value=pad_val)
         pred_bg_phase = torch.cat((pred_bg_phase, bg_phase[...,-1:]), axis=2)

         # Calculate fg/bg loss (mag. + phase)
         fg_loss = sum_loss(L1_loss, pred_fg_mag, pred_fg_phase, fg_mag, fg_phase, use_norm, phase_weight)
         bg_loss = sum_loss(L1_loss, pred_bg_mag, pred_bg_phase, bg_mag, bg_phase, use_norm, phase_weight)

         # Calculate total loss (fg loss + bg loss)
         if PIT:
            fg_loss_ = sum_loss(L1_loss, pred_fg_mag, pred_fg_phase, bg_mag, bg_phase, use_norm, phase_weight)
            bg_loss_ = sum_loss(L1_loss, pred_bg_mag, pred_bg_phase, fg_mag, fg_phase, use_norm, phase_weight)
            total_loss = torch.min(fg_loss + bg_loss, fg_loss_ + bg_loss_)
         else:
            total_loss = fg_loss + bg_loss

         # Update model
         if is_training:
            total_loss.backward()
            optimizer.step()

         if not is_training and not use_norm:
            batch_snds = ret['fg_samples'].permute(0,2,1)
            batch_snds = batch_snds.numpy()
               
            gt = batch_snds.mean(axis=2)

            fg_wav = istft(pred_fg_mag, pred_fg_phase, window)
            fg_wav = fg_wav.cpu().detach().numpy()

            if PIT:
               bg_wav = istft(pred_bg_mag, pred_bg_phase, window)
               bg_wav = bg_wav.cpu().detach().numpy()

            for i in range(gt.shape[0]):
               #fg_snd = fg_wav[i] / np.max(np.abs(fg_wav[i]))
               #fg_snd = np.clip(fg_wav[i], -1., 1.)
               fg_snd = fg_wav[i]
               gt_snd = gt[i,:fg_snd.shape[-1]]
               np.save(f'result/unet-set123/fg_{file_cnt}.npy', fg_snd)
               np.save(f'result/unet-set123/gt_{file_cnt}.npy', gt_snd)
               if PIT:
                  #bg_snd = bg_wav[i] / np.max(np.abs(bg_wav[i]))
                  #bg_snd = np.clip(bg_wav[i], -1., 1.)
                  bg_snd = bg_wav[i]
                  np.save(f'result/unet-set123/bg_{file_cnt}.npy', bg_snd)
               file_cnt += 1


         fg_loss_hist.append(fg_loss.item())
         bg_loss_hist.append(bg_loss.item())
         total_loss_hist.append(total_loss.item())

         pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
         pbar.set_postfix(fg_loss=np.around(fg_loss.item(), 4), bg_loss=np.around(bg_loss.item(),4), total_loss=np.around(total_loss.item(),4))

      hist_loss['fg_loss'][epoch] = np.mean(fg_loss_hist)
      hist_loss['bg_loss'][epoch] = np.mean(bg_loss_hist)
      hist_loss['total_loss'][epoch] = np.mean(total_loss_hist)

      if hist_loss['total_loss'][epoch] < min_loss and is_training:
         min_loss = hist_loss['total_loss'][epoch]
         best_model_wts = copy.deepcopy(model.state_dict())
         state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'min_loss': min_loss}
         if save_dir:
            torch.save(state, save_dir)

   return model, hist_loss


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('-t', '--test', help='testing mode', action='store_true')
   parser.add_argument('-v', '--valid', help='validation mode', action='store_true')
   parser.add_argument('-r', '--resume', help='resume training', type=int, default=0)
   parser.add_argument('-m', '--model', help='model name', type=str, default='')
   parser.add_argument('-s', '--set', help='noise set', type=int, default=0)
   parser.add_argument('-a', '--av_noise', help='use external noise source', default=False, action='store_true')

   args = parser.parse_args()

   total_epochs = 30
   batch_size = 8
   train_path = '../data/obamanet/train/'
   valid_path = '../data/obamanet/test/'

   assert (args.set != 0), 'noise set is not given!'

   if args.av_noise:
      noise_path = f'/home/ubuntu/eecs504/multisensory/data/noise_set/noise_{args.set}/'
      print('Take the nosie from noise dataset:', noise_path)
   else:
      noise_path = None
      print('Self Generating Noise')

   if args.model.lower() == 'unet':
      model_name = f'./model/UNetModel_set{args.set}_{args.resume}.pth'
      save_dir = f'./model/UNetModel_set{args.set}_{args.resume+1}.pth'
      use_full = False
   elif args.model.lower() == 'vidsep':
      model_name = f'./model/VidSepModel_set{args.set}_{args.resume}.pth'
      save_dir = f'./model/VidSepModel_set{args.set}_{args.resume+1}.pth'
      use_full = True
   else:
      print('Undefiend model name')
      raise NotImplementedError

   vid_transform = transforms.Compose([
                     RandomHorizontalFlip(p=0.5),
                     VidToTensor()
                   ])

   snd_transform = transforms.Compose([
                     NormalizeRms(),
                     SndToTensor(),
                     MixSound(av_noise=args.av_noise)
                   ])

   net = VidSepNet(Full=use_full).to(device)

   if args.test:
      print('Testing mode: Generate MP4 files')
      checkpoint = torch.load(model_name)
      net.load_state_dict(checkpoint['state_dict'])

      #ret = predict_vidsep('/home/ubuntu/eecs504/obamanet/data/videos/00112.mp4', net, noise_path)
      ret = predict_vidsep('../data/translator.mp4', net, None)

      print("fg_wav shape", ret['pred_fg'].shape)
      print("bg_wav shape", ret['pred_bg'].shape)
      out_dir = './results/'

      wavfile.write(out_dir+'fg.wav', int(samp_sr), ret['pred_fg'].reshape(-1,))
      wavfile.write(out_dir+'bg.wav', int(samp_sr), ret['pred_bg'].reshape(-1,))
      wavfile.write(out_dir+'orig.wav', int(samp_sr), ret['source'].mean(axis=1))
      try:
         wavfile.write(out_dir+'noise.wav', int(samp_sr), ret['noise'].reshape(-1,))
         wavfile.write(out_dir+'mix_sound.wav', int(samp_sr), ret['mix_sound'].mean(axis=1))
      except:
         print('no intentional background noise')

      print("Finished!")
      
   elif args.valid:
      assert (duration_mult == 1.), 'duration_mult in params.py should be 1'
      print('Validation mode: Evaluate the validation set')
      dataset = VidSepDataset(valid_path+'frames/', valid_path+'audios/', noise_path, vid_transform, snd_transform)
      dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=6)

      print('Total Validation data', len(dataset))
      checkpoint = torch.load(model_name)
      net.load_state_dict(checkpoint['state_dict'])
      model, val_loss2  = train(net, dataloader, optimizer=None, num_epochs=1, 
                                       save_dir=save_dir, use_norm=False, is_training=False)

      '''
      model, val_loss1  = train(net, dataloader, optimizer=None, num_epochs=1, 
                                 save_dir=save_dir, use_norm=True, is_training=False)

      print('Validation total loss with norm the range', np.mean(val_loss1['total_loss']))
      '''
      print('Validation total loss without norm the range', np.mean(val_loss2['total_loss']))

   else:
      assert (duration_mult == 1.), 'duration_mult in params.py should be 1'

      train_dataset = VidSepDataset(train_path+'frames/', train_path+'audios/', noise_path, vid_transform, snd_transform)
      data_dict = {'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6)}

      print("Total Training data", len(train_dataset))

      if args.resume == 0:
         print('Training from scratch!')
         net.weight_init(mean=0.0, std=0.02)
         optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-3)

         model, hist_loss = train(net, data_dict['train'], optimizer, use_norm=False,
                                  num_epochs=total_epochs, save_dir=save_dir, PIT=True)
      else:
         print(f'Loading a checkpoint from a pre-trained model of  Epoch: {args.resume}')
         checkpoint = torch.load(model_name)
         net.load_state_dict(checkpoint['state_dict'])
         optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
         optimizer.load_state_dict(checkpoint['optimizer'])
         min_loss = checkpoint['min_loss']

         model, hist_loss = train(net, data_dict['train'], optimizer, use_norm=False, 
                                    num_epochs=total_epochs, save_dir=save_dir, PIT=True)

