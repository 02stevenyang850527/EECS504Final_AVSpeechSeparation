import multiprocess
from collections import namedtuple

import numpy as np
import librosa

from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader

class VideoNormalizer(object):

	def __init__(self, video_samples):
		# video_samples: slices x height x width x frames_per_slice
		self.__mean_image = np.mean(video_samples, axis=(0, 3))
		self.__std_image = np.std(video_samples, axis=(0, 3))

	def normalize(self, video_samples):
		for s in range(video_samples.shape[0]):
			for f in range(video_samples.shape[3]):
				video_samples[s, :, :, f] -= self.__mean_image
				video_samples[s, :, :, f] /= self.__std_image