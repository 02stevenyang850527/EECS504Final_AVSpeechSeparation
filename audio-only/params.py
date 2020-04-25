import numpy as np


start = 0.         # how many seconds into the video to start 

duration_mult = 4. # Multiply the default duration of the audio (i.e.
                   # 2.135000) by this amount. Should be a power of 2.
VidDur = 2.135
total_dur = 5.
fps = 29.97
frame_dur = 1./fps
l1_weight = 1.
phase_weight = 0.01
gan_weight = 0.
spec_min = -100.
spec_max = 80.
frame_length_ms = 64
frame_step_ms = 16
sample_len = None
freq_len = 1024
pit_weight = 0.
samp_sr = 21000.  

full_im_dim = 256
crop_im_dim = 224

if duration_mult != 1:
   step = 0.001 * frame_step_ms
   length = 0.001 * frame_length_ms
   clip_dur = length + step*(0.5+128)*duration_mult
else:
   clip_dur = VidDur

full_dur = clip_dur + 0.01
step_dur = clip_dur / 2.

stft_frame_length = int(frame_length_ms * samp_sr * 0.001)
stft_frame_step   = int(frame_step_ms * samp_sr * 0.001)
stft_num_fft      = int(2**np.ceil(np.log2(stft_frame_length)))

vid_dur = VidDur if duration_mult == 1 else clip_dur

total_frames = int(total_dur*fps)
sampled_frames = int(vid_dur*fps)
full_samples_len = int(total_dur * samp_sr)
samples_per_frame = samp_sr * frame_dur
frame_sample_delta = int(total_dur*fps)/2

spec_len = 128 * int(2**np.round(np.log2(vid_dur/float(VidDur))))
num_samples = int(round(samples_per_frame*sampled_frames))

