from keras import backend as K
from keras.engine import Layer
import numpy as np
import os

def prepareX(X, CHOP_SIZE, MAX_SENT_LENGTH):
    MAX_SENTS=MAX_SENT_LENGTH/CHOP_SIZE
    data = np.zeros((len(X), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    for i, doc in enumerate(X):
        for j, sent in enumerate(doc):
            if j< MAX_SENTS:
                k=1
                for word in reversed(sent):
                    if k<=MAX_SENT_LENGTH:
                        data[i,j,-k] = word
                        k=k+1
    return data

def reverse_func(x):
    # For theano back-end:
    # if os.environ['KERAS_BACKEND'] == 'theano':
    #     return x[:,::-1,:]
    #
    # # For tensorflow back-end:
    # if os.environ['KERAS_BACKEND'] == 'tensorflow':
    return K.reverse(x, [1])

class OneHot(Layer):
    '''Turn positive integers (indexes) into dense dummy vectors of fixed size
    eg. [[0], [1], [2]] -&gt; [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    This layer can only be used as the first layer in a model.

    # Example

    ```python
        model = Sequential()
        model.add(OneHot(3))
        # now: model.output_shape == (None, 1, 3)

    # Arguments
        input_dim: int &gt; 0. Size of the vocabulary, ie.

    # Input shape
        2D tensor with shape: `(nb_samples, sequence_length)`.

    # Output shape
        3D tensor with shape: `(nb_samples, sequence_length, input_dim)`.
    '''

    def __init__(self, input_dim, input_length=None, **kwargs):
        self.input_dim = input_dim
        self.input_length = input_length
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = 'int32'
        super(OneHot, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        return input_shape[0], input_length, self.input_dim

    def call(self, x, mask=None):
        return K.one_hot(x, self.input_dim)

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(OneHot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
