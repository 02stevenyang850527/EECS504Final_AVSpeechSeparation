import multiprocess
from collections import namedtuple

import numpy as np
import librosa

from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader


def preprocess_audio_signal(audio_signal, slice_duration_ms, n_video_slices, video_frame_rate):
	samples_per_slice = int((float(slice_duration_ms) / 1000) * audio_signal.get_sample_rate())
	signal_length = samples_per_slice * n_video_slices

	if audio_signal.get_number_of_samples() < signal_length:
		audio_signal.pad_with_zeros(signal_length)
	else:
		audio_signal.truncate(signal_length)

	n_fft = int(float(audio_signal.get_sample_rate()) / video_frame_rate)
	hop_length = int(n_fft / 4)

	mel_spectrogram, phase = signal_to_spectrogram(audio_signal, n_fft, hop_length, mel=True, db=True)

	spectrogram_samples_per_slice = int(samples_per_slice / hop_length)
	n_slices = int(mel_spectrogram.shape[1] / spectrogram_samples_per_slice)

	slices = [
		mel_spectrogram[:, (i * spectrogram_samples_per_slice):((i + 1) * spectrogram_samples_per_slice)]
		for i in range(n_slices)
	]

	return np.stack(slices)


Sample = namedtuple('Sample', [
	'speaker_id',
	'video_file_path',
	'speech_file_path',
	'noise_file_path',
	'video_samples',
	'mixed_spectrograms',
	'speech_spectrograms',
	'noise_spectrograms',
	'mixed_signal',
	'video_frame_rate'
])


def preprocess_sample(speech_entry, noise_file_path, slice_duration_ms=200):
	print("preprocessing sample: %s, %s, %s..." % (speech_entry.video_path, speech_entry.audio_path, noise_file_path))


	mouth_height=128
	mouth_width=128

	print("preprocessing %s" % speech_entry.video_path)

	face_detector = FaceDetector()
	a = speech_entry.video_path

	with VideoFileReader(a) as reader:

		frames = reader.read_all_frames(convert_to_gray_scale=True)

		mouth_cropped_frames = np.zeros(shape=(mouth_height, mouth_width, 75), dtype=np.float32)
		for i in range(75):
			mouth_cropped_frames[:, :, i] = face_detector.crop_mouth(frames[i], bounding_box_shape=(mouth_width, mouth_height))

		frames_per_slice = int(slice_duration_ms / 1000 *  reader.get_frame_rate())

		slices = [ mouth_cropped_frames[:, :, (i * frames_per_slice):((i + 1) * frames_per_slice)] for i in range(int(75 / frames_per_slice)) ]

		video_samples = np.stack(slices)
		video_frame_rate = reader.get_frame_rate()

	print("preprocessing pair: %s, %s" % (speech_entry.audio_path, noise_file_path))

	speech_signal = AudioSignal.from_wav_file(speech_entry.audio_path)
	print(noise_file_path)
	noise_signal = AudioSignal.from_wav_file(noise_file_path)
	print(noise_signal.get_data())
	print(noise_signal.get_sample_rate())
	noise_signal.save_to_wav_file('./noise.wav')
	while noise_signal.get_number_of_samples() < speech_signal.get_number_of_samples():
		noise_signal = AudioSignal.concat([noise_signal, noise_signal])

	noise_signal.truncate(speech_signal.get_number_of_samples())

	factor = AudioMixer.snr_factor(speech_signal, noise_signal, snr_db=0)
	# print(factor)
	noise_signal.amplify_by_factor(factor)
	
	
	#noise_signal.save_to_wav_file('./noise.wav')
	mixed_signal = AudioMixer.mix([speech_signal, noise_signal], mixing_weights=[1, 1])
	mixed_signal.save_to_wav_file('./mixed.wav')
	mixed_spectrograms = preprocess_audio_signal(mixed_signal, slice_duration_ms, video_samples.shape[0], video_frame_rate)
	speech_spectrograms = preprocess_audio_signal(speech_signal, slice_duration_ms, video_samples.shape[0], video_frame_rate)
	noise_spectrograms = preprocess_audio_signal(noise_signal, slice_duration_ms, video_samples.shape[0], video_frame_rate)


	n_slices = min(video_samples.shape[0], mixed_spectrograms.shape[0])

	return Sample(
		speaker_id=speech_entry.speaker_id,
		video_file_path=speech_entry.video_path,
		speech_file_path=speech_entry.audio_path,
		noise_file_path=noise_file_path,
		video_samples=video_samples[:n_slices],
		mixed_spectrograms=mixed_spectrograms[:n_slices],
		speech_spectrograms=speech_spectrograms[:n_slices],
		noise_spectrograms=noise_spectrograms[:n_slices],
		mixed_signal=mixed_signal,
		video_frame_rate=video_frame_rate
	)


def try_preprocess_sample(sample_paths):
	try:
		return preprocess_sample(*sample_paths)

	except Exception as e:
		print("failed to preprocess %s (%s)" % (sample_paths, e))
		return None


def preprocess_data(speech_entries, noise_file_paths):
	print("preprocessing data...")

	sample_paths = zip(speech_entries, noise_file_paths)

	thread_pool = multiprocess.Pool(16)
	samples = thread_pool.map(try_preprocess_sample, sample_paths)
	samples = [p for p in samples if p is not None]

	return samples



