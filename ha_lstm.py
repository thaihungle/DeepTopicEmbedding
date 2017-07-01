import numpy as np
from collections import defaultdict
import re
import sys
import os
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


os.environ['KERAS_BACKEND'] = 'tensorflow'
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
import logging
import math
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import nn_topic_pretrain as ntp

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT, filename=os.getcwd() + '/test.txt', filemode='w')
logger.setLevel(logging.INFO)

MAX_SENT_LENGTH = 500
MAX_SENTS = 128
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
LSTM_DIM = 128
SMALL_SENTS = 1
MAX_NB_WORDS=500

GLOVE_DIR = "./glove"






# (X_train, y_train), (X_test, y_test), embedding_weights = ntp.get_imdb_and_emb(MAX_NB_WORDS,'./glove/')
(X_train, y_train), (X_test, y_test), embedding_weights = ntp.get_imdb_and_emb(MAX_NB_WORDS,'./glove/')
E_DIM = embedding_weights.shape[1]

embedding_layer = Embedding(embedding_weights.shape[0],
							E_DIM,
							weights=[embedding_weights],
							input_length=MAX_SENT_LENGTH,
							trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(100))(review_encoder)
preds = Dense(2, activation='softmax')(l_lstm_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical LSTM")
print
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)
