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
import keras.optimizers as opt

import nn_topic_pretrain as ntp

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT, filename=os.getcwd() + '/test.txt', filemode='w')
logger.setLevel(logging.INFO)

MAX_SENT_LENGTH = 500
MAX_SENTS = 128
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
LSTM_DIM = 50
SMALL_SENTS = 1
MAX_NB_WORDS=30000

BASE_DIR = './embfiles'
GLOVE_DIR = BASE_DIR + '/glove/'

K.set_learning_phase(1)

def build_model(model_name='ha_lstm', conti=True):
    fname_model = os.getcwd() + "/" + model_name + '/modelfile.json'
    model_dir = os.getcwd() + "/" + model_name

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    print('load prep data...')
    ((X_train, y_train),
     (X_test, y_test),
     word_index,
     (MAX_NB_WORDS,MAX_SENTS,MAX_SENT_LENGTH))\
        =	pickle.load(open('./data/imdb_prep_stem.pkl', 'rb'))

    emb_matrix=ntp.get_glove_emb_100(GLOVE_DIR,word_index,MAX_NB_WORDS)
    emb_matrix2 = ntp.get_topic_emb('./embfiles/fstm.30000.10.ha2.beta')
    # emb_matrix3 = ntp.get_topic_emb('./embfiles/fstm.30000.0.ha.beta')

    print('start build model')

    if os.path.isfile(fname_model) and conti:
        json_file = open(fname_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    else:
        sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

        embedding_layer = Embedding(emb_matrix.shape[0],
                                    emb_matrix.shape[1],
                                    weights=None,
                                    input_length=MAX_SENT_LENGTH,
                                    trainable=True)

        embedding_layer2 = Embedding(emb_matrix2.shape[0],
                                     emb_matrix2.shape[1],
                                    weights=None,
                                    input_length=MAX_SENT_LENGTH,
                                    trainable=True)



        embedded_sequences2 = embedding_layer2(sentence_input)

        embedded_sequences = embedding_layer(sentence_input)

        mv_vector = merge([embedded_sequences2, embedded_sequences], mode='concat')

        conv = Convolution1D(nb_filter=128,
                             filter_length=11,
                             border_mode='same',
                             activation='relu')(embedded_sequences2)

        mp = MaxPooling1D(pool_length=conv._keras_shape[1])(conv)
        mp = Flatten()(Dropout(0.1)(mp))
        #l_lstm2 = Bidirectional(LSTM(LSTM_DIM, return_sequences=False))(mp)

        # mv_vector = merge([conv, embedded_sequences], mode='concat')

        # dr = Dropout(0.5)(mp)

        # sentence_input2 = Masking(mask_value=0,
        #                           input_shape=(MAX_SENT_LENGTH, EMBEDDING_DIM)
        #                           )(sentence_input)

        l_lstm = Bidirectional(LSTM(LSTM_DIM,return_sequences=True))(embedded_sequences)
        att = TimeDistributed(Dense(LSTM_DIM, activation='relu'))(l_lstm)  # [n_samples, n_steps, rnn_dim]
        att = TimeDistributed(Dense(1, bias=False))(att)  # [n_samples, n_steps, 1]
        att = Flatten()(att)  # [n_samples, n_steps]
        att = Activation('softmax')(att)  # [n_samples, n_steps]
        print(att._keras_shape)
        att = Reshape((1, att._keras_shape[1]))(att)
        lstm = merge([att, l_lstm], mode='dot', dot_axes=(2, 1))  # [n_samples, rnn_dim]
        lstm = Flatten()(lstm)
        lstm = merge([lstm, mp], mode='concat')
        sentEncoder = Model(sentence_input, lstm)

        # sentEncoder = Model(sentence_input, l_lstm)

        print(sentEncoder.summary())
        review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(LSTM(LSTM_DIM, return_sequences=True))(review_encoder)
        att2 = TimeDistributed(Dense(LSTM_DIM, activation='relu'))(l_lstm_sent)  # [n_samples, n_steps, rnn_dim]
        att2 = TimeDistributed(Dense(1, bias=False))(att2)  # [n_samples, n_steps, 1]
        att2 = Flatten()(att2)  # [n_samples, n_steps]
        att2 = Activation('softmax')(att2)  # [n_samples, n_steps]
        att2 = Reshape((1, MAX_SENTS))(att2)
        lstm2 = merge([att2, l_lstm_sent], mode='dot', dot_axes=(2, 1))  # [n_samples, rnn_dim]
        print('-----------')
        print(lstm2._keras_shape)
        # lstm3 = Reshape((MAX_SENTS, -1))(lstm2)
        preds = Dense(2, activation='softmax')(Flatten()(lstm2))
        # preds = Dense(2, activation='softmax')(l_lstm_sent)
        model = Model(review_input, preds)

        with open(fname_model, "w") as json_file:
            model_json = model.to_json()
            json_file.write(model_json)

    fname_weights = os.getcwd() + "/" + model_name + "/modelweights.h5"

    if os.path.isfile(fname_weights) and conti:
        # load weights into new model
        model.load_weights(fname_weights)
        print("Loaded weights from disk")



    model.compile(loss='categorical_crossentropy',
                  optimizer= opt.RMSprop(),
                  metrics=['acc'])

    print(model.summary())

    print("model fitting - Hierachical LSTM")

    checkpointer = ModelCheckpoint(filepath=fname_weights,
                                       verbose=1,
                                       save_best_only=True)
    history = model.fit(X_train, y_train, nb_epoch=50,
        batch_size=15, validation_data=(X_test, y_test),
                            callbacks=[checkpointer])

def evaluate_model(model_name):
    model_dir = os.getcwd() + "/" + model_name
    fname_model = model_dir + '/modelfile.json'
    fname_weights = model_dir + "/modelweights.h5"


    ((X_train, y_train),
     (X_test, y_test),
     word_index,
     (MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH)) \
        = pickle.load(open('./data/imdb_prep_stem.pkl', 'rb'))

    json_file = open(fname_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(fname_weights)
    print("Loaded weights from disk")

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # Final evaluation of the model
    print("predict....")
    scores = model.evaluate(X_test, y_test, verbose=1)
    print(scores)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    model_name = './models/ha_lstm'
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
    elif mode == 'eval':
        evaluate_model(model_name)
