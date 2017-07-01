
import numpy as np
from collections import defaultdict
import re
import sys
import os
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers.core import Activation, Masking, Reshape
from keras.layers import merge
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import logging
import math
import nn_topic_pretrain as ntp

MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 500
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

BASE_DIR = './embfiles'
GLOVE_DIR = BASE_DIR + '/glove/'

def build_model(model_name='cnnlstm', conti=True):
	fname_model = os.getcwd() + "/" + model_name + '/modelfile.json'
	model_dir = os.getcwd() + "/" + model_name

	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)

	# fix random seed for reproducibility
	np.random.seed(7)
	# load the dataset but only keep the top n words, zero the rest
	print('Start load imdb data...')
	# (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=MAX_NB_WORDS)

	# print(X_train[0])
	# print('===========================')
	# print(y_train[0])
	#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=MAX_NB_WORDS)
	(X_train, y_train), (X_test, y_test), embedding_weights = ntp.get_imdb_and_emb(MAX_NB_WORDS,GLOVE_DIR)
	# embedding_weights=ntp.get_topic_emb('./embfiles/fstm.5000.beta')
	E_DIM = embedding_weights.shape[1]
	#E_DIM=32
	print(X_train[0])
	print('===========================')
	print(y_train[0])
	print('Done load imdb data!')


	print ('num train samples: {}'.format(len(y_train)))
	print ('num test samples: {}'.format(len(y_test)))
	# truncate and pad input sequences

	X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
	X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
	y_train = to_categorical(np.asarray(y_train))
	y_test = to_categorical(np.asarray(y_test))
	# num_validation_samples = int(VALIDATION_SPLIT * len(X_train))
	#
	# X_tr = X_train[:-num_validation_samples]
	# y_tr = y_train[:-num_validation_samples]
	# x_val = X_train[-num_validation_samples:]
	# y_val = y_train[-num_validation_samples:]

	# create the model


	if os.path.isfile(fname_model) and conti:
		json_file = open(fname_model, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
	else:
		print('build model...')
		embedding_layer = Embedding(embedding_weights.shape[0],
								E_DIM,
								weights=[embedding_weights],
								input_length=MAX_SEQUENCE_LENGTH,
								trainable=False)

		sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
		embedded_sequences = embedding_layer(sequence_input)
		conv = Convolution1D(nb_filter=128,
							 filter_length=9,
							 border_mode='same',
							 activation='relu')(embedded_sequences)

		mp = MaxPooling1D(pool_length=5)(conv)
		dr = Dropout(0.5)(mp)
		l_lstm = Bidirectional(LSTM(128, return_sequences=False
								   # W_regularizer=regularizers.l2(0.1),
								   # U_regularizer=regularizers.l2(0.1)
								   ))(dr)
		preds = Dense(2, activation='softmax')(l_lstm)
		model = Model(sequence_input, preds)

		with open(fname_model, "w") as json_file:
			model_json = model.to_json()
			json_file.write(model_json)

	fname_weights = os.getcwd() + "/" + model_name + "/modelweights.h5"

	if os.path.isfile(fname_weights) and conti:
		# load weights into new model
		model.load_weights(fname_weights)
		print("Loaded weights from disk")

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	print('done build!')
	print('start training...')
	print('start training...')
	checkpointer = ModelCheckpoint(filepath=fname_weights,
								   verbose=1,
								   save_best_only=True)


	history = model.fit(X_train, y_train, nb_epoch=50,
	batch_size=100, validation_data=(X_test, y_test),
						callbacks=[checkpointer])

if __name__ == '__main__':
	model_name = './models/cnnlstm'
	con = True
	mode = 'build'
	if len(sys.argv) >= 2:
		print("has model name")
		model_name = sys.argv[1]
	if len(sys.argv) >= 3:
		print("has con")
		con = sys.argv[2]
		if con == "True":
			con = True
		else:
			con = False
	if len(sys.argv) >= 4:
		mode = sys.argv[3]
	if mode == 'build':
		build_model(model_name, con)



