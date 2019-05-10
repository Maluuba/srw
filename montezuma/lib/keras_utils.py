from keras import backend as K
import numpy as np


def clipped_sum_error(y_true, y_pred):
    """
    Gradient Cipping as in DQN Paper
    """
    errs = y_pred - y_true
    quad = K.minimum(abs(errs), 1)
    lin = abs(errs) - quad
    return K.sum(0.5 * quad ** 2 + lin)


def slice_tensor_tensor(tensor, tensor_slice):
    """
        Theano and tensorflow differ in the method of extracting the value of the actions taken
        arg1: the tensor to be slice i.e Q(s)
        arg2: the indices to slice by ie a
    """
    if K.backend() == 'theano':
        output = tensor[K.T.arange(tensor_slice.shape[0]), tensor_slice]
    elif K.backend() == 'tensorflow':
        amask = K.tf.one_hot(tensor_slice, tensor.get_shape()[1], 1.0, 0.0)
        output = K.tf.reduce_sum(tensor * amask, reduction_indices=1)
    else:
        raise Exception("Not using theano or tensor flow as backend")
    return output


def keras_model_reset(model, init_method):
    """
    Usage:
        from keras import initializations
        keras_model_reset(your_model, initializations.glorot_uniform)
    for the init methods see: https://github.com/fchollet/keras/blob/master/keras/initializations.py
    """
    w1 = []
    for w in model.get_weights():
        w1.append(init_method(shape=w.shape).get_value())
    model.set_weights(w1)
