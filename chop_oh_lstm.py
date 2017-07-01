import os
os.environ['KERAS_BACKEND'] = 'theano'
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
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, ChainCRF
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations
from keras.layers.core import TimeDistributedDense, Activation, Masking
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.layers import merge

GRU_DIM=500
TOP_DROP_OUT=0.5
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 10000
print('start load imdb...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
print('done load imdb!')
max_review_length = 500
CHOP_SIZE=50
print('start padding...')
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

X_train, y_train = ol.prepareXychop(X_train, y_train, CHOP_SIZE)

y_train = to_categorical(numpy.asarray(y_train))
y_test = to_categorical(numpy.asarray(y_test))
print('done padding!')
# create the model

print('build model...')
review_input = Input(shape=(CHOP_SIZE,), dtype='int32')
review_encoder = ol.OneHot(top_words+1, input_length=None)(review_input)
l_gru = GRU(GRU_DIM,return_sequences=True)
train_lgru=l_gru(review_encoder)
mp1=MaxPooling1D(CHOP_SIZE)(train_lgru)
l_gru2 = GRU(GRU_DIM,return_sequences=True, go_backwards = True)
train_lgru2=l_gru2(review_encoder)
mp2=MaxPooling1D(CHOP_SIZE)(train_lgru2)
mp=merge([mp1, mp2], mode='concat')
l_flat = Flatten()(mp)
preds = Dense(2, activation='softmax')
preds_train = preds(Dropout(TOP_DROP_OUT)(l_flat))
model = Model(review_input, preds_train)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
metrics=['acc'])
print(model.summary())

review_input_test = Input(shape=(max_review_length,), dtype='int32')
review_encoder_test = ol.OneHot(top_words+1, input_length=None)(review_input)
test_lgru=l_gru(review_encoder_test)
mp1=MaxPooling1D(max_review_length)(test_lgru)
test_lgru2=l_gru2(review_encoder_test)
mp2=MaxPooling1D(max_review_length)(test_lgru2)
mp=merge([mp1, mp2], mode='concat')
l_flat = Flatten()(mp)
preds_train = preds(Dropout(TOP_DROP_OUT)(l_flat))
model2 = Model(review_input, preds_train)
model2.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
metrics=['acc'])
print(model2.summary())
print('done build!')
print('start training...')
model.fit(X_train, y_train, nb_epoch=1, batch_size=10)
# Final evaluation of the model
print('start evaluating...')
scores = model2.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
