
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
import onehot_layer as ol

MAX_SEQUENCE_LENGTH = 2000
MAX_NB_WORDS = 30000
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
    #(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=MAX_NB_WORDS)

    # ((X_train, y_train),
    #  (X_test, y_test),
    #  word_index,
    #  (MAX_NB_WORDS, MAX_SENT_LENGTH)) \
    #     = pickle.load(open('./data/imdb_prep_context.pkl', 'rb'))

    (X_train, y_train), (X_test, y_test), embedding_weights \
        = ntp.get_imdb_and_emb(MAX_NB_WORDS,GLOVE_DIR)

    y_train=np.asarray(to_categorical(y_train))
    y_test = np.asarray(to_categorical(y_test))

    #embedding_weights=ntp.get_topic_emb('./embfiles/fstm.30000.60.beta')

    # embedding_weights = ntp.get_glove_emb_100(GLOVE_DIR, word_index, MAX_NB_WORDS)
    embedding_weights2=ntp.get_topic_emb('./embfiles/fstm.30000.60.beta')

    print(X_train[0])
    print('===========================')
    print(y_train[0])
    print('Done load imdb data!')


    print ('num train samples: {} {}'.format(len(y_train), len(X_train)))
    print ('num test samples: {} {}'.format(len(y_test), len(X_test)))
    # truncate and pad input sequences

    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)



    if os.path.isfile(fname_model) and conti:
        json_file = open(fname_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
    else:
        print('build model...')

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')



        embedding_layer = Embedding(embedding_weights2.shape[0],
                                    embedding_weights2.shape[1],
                                    weights=[embedding_weights2],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)

        embedded_sequences2 = embedding_layer(sequence_input)


        embedding_layer = Embedding(embedding_weights.shape[0],
                                    embedding_weights.shape[1],
                                weights=[embedding_weights],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)


        embedded_sequences = embedding_layer(sequence_input)

        mv_vector = merge([embedded_sequences2, embedded_sequences], mode='concat')

        conv = Convolution1D(nb_filter=128,
                             filter_length=9,
                             border_mode='same',
                             activation='relu')(mv_vector)

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

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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

def evaluate_model(model_name):
    model_dir = os.getcwd() + "/" + model_name
    fname_model = model_dir + '/modelfile.json'
    fname_weights = model_dir + "/modelweights.h5"
    # load json and create model

    print('start load imdb...')


    (X_train, y_train), (X_test, y_test) \
    = ntp.get_imdb_and_emb(MAX_NB_WORDS, GLOVE_DIR, 0)
    print('done load imdb!')

    print('start padding...')
    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = to_categorical(np.asarray(y_train))
    y_test = to_categorical(np.asarray(y_test))

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
    elif mode == 'eval':
        evaluate_model(model_name)




