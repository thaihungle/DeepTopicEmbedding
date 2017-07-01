import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import numpy
import onehot_layer as ol
import sys


from keras.datasets import imdb
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
from keras.layers.core import  Activation, Masking, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.layers import merge

import nn_topic_pretrain as ntp

def build_model(model_name='mv_lstm', conti=True):
    fname_model = os.getcwd() + "/" + model_name + '/modelfile.json'
    model_dir = os.getcwd() + "/" + model_name

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    print('start load imdb...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
    print('done load imdb!')
    max_review_length = 500
    CHOP_SIZE=50
    print('start padding...')
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # X_train, y_train = ol.prepareXychop(X_train, y_train, CHOP_SIZE)

    y_train = to_categorical(numpy.asarray(y_train))
    y_test = to_categorical(numpy.asarray(y_test))
    print('done padding!')
    # create the model
    if os.path.isfile(fname_model) and conti:
        json_file = open(fname_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json,{'OneHot': ol.OneHot})
    else:
        print('build model...')
        review_input = Input(shape=(max_review_length,), dtype='int32')

        review_encoder = ol.OneHot(top_words+1, input_length=max_review_length)(review_input)
        embedding_weights=ntp.get_topic_emb('./embfiles/fstm.5000.beta')
        print(embedding_weights.shape)

        embedding_layer = Embedding(top_words,
                                            embedding_weights.shape[1],
                                            weights=[embedding_weights],
                                            input_length=max_review_length,
                                            trainable=True
                                            )

        embedded_sequences = embedding_layer(review_input)


        mv_vector=merge([review_encoder,embedded_sequences],mode='concat')

        l_gru = GRU(100,return_sequences=True)(mv_vector)
        mp1=MaxPooling1D(100)(l_gru)

        l_gru2 = GRU(100,return_sequences=True, go_backwards=True)(mv_vector)
        l_gru2 = Lambda(ol.reverse_func)(l_gru2)
        print(l_gru2.shape)
        mp2=MaxPooling1D(100)(l_gru2)
        mp=merge([mp1, mp2], mode='concat')
        l_flat = Flatten()(mp)
        preds = Dense(2, activation='softmax')(Dropout(0.5)(l_flat))
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
                  optimizer='rmsprop',
    metrics=['acc'])
    print(model.summary())
    print('done build!')
    print('start training...')
    print('start training...')
    checkpointer = ModelCheckpoint(filepath=fname_weights,
                                   verbose=1,
                                   save_best_only=True)

    model.fit(X_train, y_train,
              nb_epoch=100,
              batch_size=50,
              validation_data=(X_test, y_test),
              callbacks=[checkpointer])

    # Final evaluation of the model
    print('start evaluating...')
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))


if __name__ == '__main__':
    model_name='./models/oh_lstm'
    con=True
    mode='build'
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
    if mode=='build':
        build_model(model_name,con)