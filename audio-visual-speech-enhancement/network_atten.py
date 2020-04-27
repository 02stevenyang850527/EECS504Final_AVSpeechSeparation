from keras import optimizers
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape
from keras.layers import Bidirectional, LSTM
from keras.layers import Activation, RepeatVector, Permute, multiply
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import numpy as np


class SpeechEnhancementNetwork(object):

	def __init__(self, model):
		self.__model = model

############################
	@classmethod
	def build(cls, audio_spectrogram_shape, video_shape):
		# append channels axis
		extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
		extended_audio_spectrogram_shape.append(1)

		encoder, shared_embedding_size, audio_embedding_shape = cls.__audio_video_encoder(extended_audio_spectrogram_shape, video_shape)
		decoder = cls.__decoder(shared_embedding_size, audio_embedding_shape)

		audio_input = Input(shape=extended_audio_spectrogram_shape)
		video_input = Input(shape=video_shape)

		audio_output = decoder(encoder([audio_input, video_input]))

		model = Model(inputs=[audio_input, video_input], outputs=audio_output)

		optimizer = optimizers.adam(lr=5e-4)
		model.compile(loss='mean_squared_error', optimizer=optimizer)

		model.summary()

		return SpeechEnhancementNetwork(model)
########################
	def __audio_video_encoder(extended_audio_spectrogram_shape, video_shape):
		audio_input = Input(shape=extended_audio_spectrogram_shape)
		video_input = Input(shape=video_shape)
		
		aux = Convolution2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(audio_input)
		aux = BatchNormalization()(aux)
		aux = LeakyReLU()(aux)

		aux = Convolution2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(aux)
		aux = BatchNormalization()(aux)
		aux = LeakyReLU()(aux)

		aux = Convolution2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(aux)
		aux = BatchNormalization()(aux)
		aux = LeakyReLU()(aux)

		aux = Convolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(aux)
		aux = BatchNormalization()(aux)
		aux = LeakyReLU()(aux)

		aux = Convolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(aux)
		aux = BatchNormalization()(aux)
		aux_out = LeakyReLU()(aux)
		aux_out_size = aux_out._keras_shape
		## for temporal layer
		aux = Reshape((-1, aux_out_size[-1]))(aux_out)
		#aux = Flatten()(aux_out)


		vix = Convolution2D(128, kernel_size=(5, 5), padding='same')(video_input)
		vix = BatchNormalization()(vix)
		vix = LeakyReLU()(vix)
		vix = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(vix)
		vix = Dropout(0.25)(vix)

		vix = Convolution2D(128, kernel_size=(5, 5), padding='same')(vix)
		vix = BatchNormalization()(vix)
		vix = LeakyReLU()(vix)
		vix = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(vix)
		vix = Dropout(0.25)(vix)

		vix = Convolution2D(256, kernel_size=(3, 3), padding='same')(vix)
		vix = BatchNormalization()(vix)
		vix = LeakyReLU()(vix)
		vix = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(vix)
		vix = Dropout(0.25)(vix)

		vix = Convolution2D(256, kernel_size=(3, 3), padding='same')(vix)
		vix = BatchNormalization()(vix)
		vix = LeakyReLU()(vix)
		vix = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(vix)
		vix = Dropout(0.25)(vix)

		vix = Convolution2D(512, kernel_size=(3, 3), padding='same')(vix)
		vix = BatchNormalization()(vix)
		vix = LeakyReLU()(vix)
		vix = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(vix)
		vix = Dropout(0.25)(vix)

		vix = Convolution2D(512, kernel_size=(3, 3), padding='same')(vix)
		vix = BatchNormalization()(vix)
		vix = LeakyReLU()(vix)
		vix = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(vix)
		vix = Dropout(0.25)(vix)
		
		vix = Reshape((-1, aux_out_size[-1]))(vix)
		#vix = Flatten()(vix)

		x = concatenate([aux, vix], axis=1)			## should be [None, -1, 128]
		#shared_embedding_size = int(x._keras_shape[1] / 4)		## 1312
		shared_embedding_size = 1312

		## add an LSTM + attention layer
		x = Bidirectional(LSTM(400))(x)					## should be [None, 800]
		## Attention Calculation
		attention_prob = Dense(800, activation='relu')(x)
		x = multiply([x, attention_prob])

		x = Dense(shared_embedding_size)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		print(x.shape)
		model = Model(inputs=[audio_input, video_input], outputs=x)
		model.summary()
		return model, shared_embedding_size, aux_out.shape[1:].as_list()
######################
	@classmethod
	def __decoder(cls, shared_embedding_size, audio_embedding_shape):

		shared_embedding_input = Input(shape=(shared_embedding_size,))
		
		x = Dense(shared_embedding_size)(shared_embedding_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Dense(np.prod(audio_embedding_shape))(x)
		x = Reshape(audio_embedding_shape)(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

		model = Model(inputs=shared_embedding_input, outputs=x)
		model.summary()

		return model

#####################
	def train(self, train_mixed_spectrograms, train_video_samples, train_speech_spectrograms,
			  validation_mixed_spectrograms, validation_video_samples, validation_speech_spectrograms,
			  model_cache_path, tensorboard_dir):

		train_mixed_spectrograms = np.expand_dims(train_mixed_spectrograms, -1)  # append channels axis
		train_speech_spectrograms = np.expand_dims(train_speech_spectrograms, -1)  # append channels axis

		validation_mixed_spectrograms = np.expand_dims(validation_mixed_spectrograms, -1)  # append channels axis
		validation_speech_spectrograms = np.expand_dims(validation_speech_spectrograms, -1)  # append channels axis

		checkpoint = ModelCheckpoint(model_cache_path, verbose=1)

		lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)

		tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)

		self.__model.fit(
			x=[train_mixed_spectrograms, train_video_samples],
			y=train_speech_spectrograms,

			validation_data=(
				[validation_mixed_spectrograms, validation_video_samples],
				validation_speech_spectrograms
			),

			batch_size=16, epochs=1000,
			callbacks=[checkpoint, lr_decay, early_stopping],
			verbose=1
		)

	def predict(self, mixed_spectrograms, video_samples):
		mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
		speech_spectrograms = self.__model.predict([mixed_spectrograms, video_samples])

		return np.squeeze(speech_spectrograms)

	def evaluate(self, mixed_spectrograms, video_samples, speech_spectrograms):
		mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
		speech_spectrograms = np.expand_dims(speech_spectrograms, -1)  # append channels axis
		
		loss = self.__model.evaluate(x=[mixed_spectrograms, video_samples], y=speech_spectrograms)

		return loss

	@staticmethod
	def load(model_cache_path):
		model = load_model(model_cache_path)

		return SpeechEnhancementNetwork(model)

	def save(self, model_cache_path):
		self.__model.save(model_cache_path)
