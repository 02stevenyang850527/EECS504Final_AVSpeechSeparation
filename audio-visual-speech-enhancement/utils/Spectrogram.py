import multiprocess
from collections import namedtuple

import numpy as np
import librosa

from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader

def signal_to_spectrogram(audio_signal, n_fft, hop_length, mel=True, db=True):
	signal = audio_signal.get_data(channel_index=0)
	D = librosa.core.stft(signal.astype(np.float64), n_fft=n_fft, hop_length=hop_length)
	magnitude, phase = librosa.core.magphase(D)

	if mel:
		mel_filterbank = librosa.filters.mel(
			sr=audio_signal.get_sample_rate(),
			n_fft=n_fft,
			n_mels=80,
			fmin=0,
			fmax=8000
		)

		magnitude = np.dot(mel_filterbank, magnitude)

	if db:
		magnitude = librosa.amplitude_to_db(magnitude)

	return magnitude, phase


def reconstruct_signal_from_spectrogram(magnitude, phase, sample_rate, n_fft, hop_length, mel=True, db=True):
	if db:
		magnitude = librosa.db_to_amplitude(magnitude)

	if mel:
		mel_filterbank = librosa.filters.mel(
			sr=sample_rate,
			n_fft=n_fft,
			n_mels=80,
			fmin=0,
			fmax=8000
		)

		magnitude = np.dot(np.linalg.pinv(mel_filterbank), magnitude)

	signal = librosa.istft(magnitude * phase, hop_length=hop_length)

	return AudioSignal(signal, sample_rate)