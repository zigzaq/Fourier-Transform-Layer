# author: Jakub Zak, IBIB PAN
# for: TF 2.0+
# usage: as typical keras layers, either Sequential add or x = ftl(params)(x)
# activation - what activation to pull from keras; available for now: None, relu, softmax, sigmoid, tanh, selu;
# recommended - None, relu or selu
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import relu, softmax, sigmoid, tanh, selu


class FTL(Layer):
    def __init__(self, activation=None, **kwargs):
        self.kernel = None
        self.kernel_imag = None
        self.activ = None
        if activation == 'relu':
            self.activ = relu
        elif activation == 'softmax':
            self.activ = softmax
        elif activation == 'sigmoid':
            self.activ = sigmoid
        elif activation == 'tanh':
            self.activ = tanh
        elif activation == 'selu':
            self.activ = selu
        self.initializer = 'he_normal'
        self.phase_training = False
        self.inverse = False
        super(FTL, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) == 2:
            self.kernel = self.add_weight(name='kernel',
                                          shape=tuple(input_shape[0][1:]),
                                          initializer=self.initializer,
                                          trainable=True)
            self.kernel_imag = self.add_weight(name='kernel_imag',
                                               shape=tuple(input_shape[0][1:]),
                                               initializer=self.initializer,
                                               trainable=True)
        else:
            self.kernel = self.add_weight(name='kernel',
                                               shape=tuple(input_shape[1:]),
                                               initializer=self.initializer,
                                               trainable=True)
            self.kernel_imag = self.add_weight(name='kernel_imag',
                                               shape=tuple(input_shape[1:]),
                                               initializer=self.initializer,
                                               trainable=True)
        super(FTL, self).build(input_shape)  # Be sure to call this at the end

    def call(self, input_tensor):
        # ifft for 2-tuple input
        if type(input_tensor) is tuple:
            x = tf.dtypes.complex(input_tensor[0] * self.kernel, input_tensor[1] * self.kernel_imag)
            if self.inverse:
                x = tf.signal.ifft3d(tf.cast(x, tf.complex64))
            x = tf.math.abs(x)
            if self.activ is not None:
                return self.activ(x)
            return x

        shapes = tf.shape(input_tensor)[1:]
        x = tf.signal.fft3d(tf.cast(input_tensor, tf.complex64))
        x = x / tf.cast((shapes[0] * shapes[1]), tf.complex64)
        real = tf.math.real(x) * self.kernel
        imag = tf.math.imag(x) * self.kernel_imag
        x = tf.cast(tf.dtypes.complex(real, imag), tf.complex64)
        if self.phase_training:
            if self.activ is not None:
                return self.activ(tf.math.real(x)), tf.math.imag(x)
            return tf.math.real(x), tf.math.imag(x)
        if self.inverse:
            x = tf.signal.ifft3d(x)
        x = tf.math.abs(x)
        if self.activ is not None:
            return self.activ(x)
        return x

    def compute_output_shape(self, input_shape):
        if self.phase_training:
            return input_shape, input_shape
        return input_shape