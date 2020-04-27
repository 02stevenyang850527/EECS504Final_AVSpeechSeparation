import multiprocess
from collections import namedtuple

import numpy as np
import librosa

from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader

def reconstruct_speech_signal(mixed_signal, speech_spectrograms, video_frame_rate):
	n_fft = int(float(mixed_signal.get_sample_rate()) / video_frame_rate)
	hop_length = int(n_fft / 4)

	_, original_phase = signal_to_spectrogram(mixed_signal, n_fft, hop_length, mel=True, db=True)

	speech_spectrogram = np.concatenate(list(speech_spectrograms), axis=1)

	spectrogram_length = min(speech_spectrogram.shape[1], original_phase.shape[1])
	speech_spectrogram = speech_spectrogram[:, :spectrogram_length]
	original_phase = original_phase[:, :spectrogram_length]

	return reconstruct_signal_from_spectrogram(
		speech_spectrogram, original_phase, mixed_signal.get_sample_rate(), n_fft, hop_length, mel=True, db=True
	)