import os
os.environ['KERAS_BACKEND']='tensorflow'

import numpy
import onehot_layer as ol

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
from keras.layers.core import  Activation, Masking
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import keras.optimizers  as opt

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 30000
print('start load imdb...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
print('done load imdb!')
max_review_length = 1000
print('start padding...')
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
y_train = to_categorical(numpy.asarray(y_train))
y_test = to_categorical(numpy.asarray(y_test))

embedding_weights=ntp.get_topic_emb('./embfiles/fstm.30000.60.beta')  


print('done padding!')
# create the model

print('build model...')
review_input = Input(shape=(max_review_length,), dtype='int32')
review_encoder = ol.OneHot(top_words+1, input_length=max_review_length)(review_input)

embedding_layer = Embedding(embedding_weights.shape[0],                                                                                │················································
                                    embedding_weights.shape[1],                                                                                │················································
                                weights=[embedding_weights],                                                                                   │················································
                                input_length=MAX_SEQUENCE_LENGTH,                                                                              │················································
                                trainable=True)                                                                                                │················································
                                                                                                                                               │················································
                                                                                                                                               │················································
embedded_sequences = embedding_layer(review_input) 

l_cov1= Conv1D(1000, 3, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(l_cov1._keras_shape[1])(l_cov1)
l_flat = Flatten()(l_pool1)
preds = Dense(2, kernel_regularizer=regularizers.l2(1e-4),
              activation='softmax')(Dropout(0.5)(l_flat))
model = Model(review_input, preds)
model.compile(loss='mean_squared_error',
              optimizer=opt.RMSprop(),
metrics=['acc'])
print(model.summary())
print('done build!')
print('start training...')
model.fit(X_train, y_train, nb_epoch=100, batch_size=40,validation_data=(X_test, y_test))
# Final evaluation of the model
print('start evaluating...')
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
