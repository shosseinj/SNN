from __future__ import print_function, absolute_import
import logging
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras import backend as K
from keras import regularizers
from utils import *
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.layers import Lambda


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                     Dropout, MaxPooling2D, Flatten, Dense)
from tensorflow.keras import regularizers

def VGG16(input_shape=(32, 32, 3), classes=10, weights_path='weights/cifar10vgg.h5'):
    weight_decay = 0.0005
    inputs = Input(shape=input_shape)

    # ---------------- Block 1 ----------------
    x = Conv2D(64, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # ---------------- Block 2 ----------------
    x = Conv2D(128, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # ---------------- Block 3 ----------------
    x = Conv2D(256, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # ---------------- Block 4 ----------------
    x = Conv2D(512, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # ---------------- Block 5 ----------------
    x = Conv2D(512, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(512, (3, 3), padding='same', activation=None,
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    # ---------------- Classifier ----------------
    x = Flatten()(x)
    x = Dense(512, activation=None, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)




    model.load_weights('weights/cifar10vgg.h5')

    return model




    import tensorflow as tf
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

# ------------------------
# Spiking call function
# ------------------------
def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    threshold = t_max - t_min - D_i
    ti = tf.matmul(tj - t_min, W) + threshold + t_min
    ti = tf.where(ti < t_max, ti, t_max)
    ti = ti + tf.random.normal(tf.shape(ti), stddev=robustness_params.get('noise', 0.0))
    return ti

# ------------------------
# Spiking layers
# ------------------------
class SpikingDense(tf.keras.layers.Layer):
    def __init__(self, units, X_n=1, outputLayer=False, robustness_params={}, input_dim=None, name=None):
        super().__init__(name=name)
        self.units = units
        self.outputLayer = outputLayer
        self.B_n = (1 + 0.5) * X_n
        self.robustness_params = robustness_params
        self.input_dim = input_dim

    def build(self, input_shape):
        if input_shape[-1] is None and self.input_dim is not None:
            input_shape = (None, self.input_dim)
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform', trainable=True, name='kernel'
        )
        self.D_i = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True, name='D_i')
        self.t_min_prev = tf.Variable(0.0, trainable=False)
        self.t_min = tf.Variable(0.0, trainable=False)
        self.t_max = tf.Variable(1.0, trainable=False)

    def call(self, tj):
        output = call_spiking(tj, self.kernel, self.D_i, self.t_min_prev, self.t_min, self.t_max, self.robustness_params)
        if self.outputLayer:
            W_mult_x = tf.matmul(self.t_min - tj, self.kernel)
            alpha = self.D_i / (self.t_min - self.t_min_prev + 1e-8)
            output = alpha * (self.t_min - self.t_min_prev) + W_mult_x
        # tf.print(self.name, "output sample:", output[0, :5])
        return output

class SpikingConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), X_n=1, padding='same', robustness_params={}, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.B_n = (1 + 0.5) * X_n
        self.robustness_params = robustness_params
        self.D_i = None
        self.t_min_prev = tf.Variable(0.0, trainable=False)
        self.t_min = tf.Variable(0.0, trainable=False)
        self.t_max = tf.Variable(1.0, trainable=False)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters),
            initializer='glorot_uniform', trainable=True, name='kernel'
        )
        self.D_i = self.add_weight(shape=(9, self.filters), initializer='zeros', trainable=True, name='D_i')

    def call(self, tj):
        padding_size = (self.kernel_size[0] // 2) if self.padding=='same' else 0
        image_size = tf.shape(tj)[1]
        tj = tf.pad(tj, [[0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0]], constant_values=self.t_min)
        tj_patches = tf.image.extract_patches(
            tj, sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1,1,1,1], rates=[1,1,1,1], padding='VALID'
        )
        W = tf.reshape(self.kernel, (-1, self.filters))
        tj_flat = tf.reshape(tj_patches, (-1, tf.shape(W)[0]))
        ti = call_spiking(tj_flat, W, self.D_i[0], self.t_min_prev, self.t_min, self.t_max, self.robustness_params)
        ti = tf.reshape(ti, (-1, image_size, image_size, self.filters))
        # tf.print(self.name, "output sample:", ti[0, :2, :2, 0])
        return ti

# ------------------------
# MaxMinPool2D
# ------------------------
class MaxMinPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
    def call(self, x):
        x = tf.nn.max_pool2d(x, ksize=self.pool_size, strides=self.pool_size, padding='SAME')
        return x

# ------------------------
# VGG-like SNN
# ------------------------
class VGG_SNN(tf.keras.Model):
    def __init__(self, layers2D, kernel_size, layers1D, data, optimizer,
                 robustness_params={}):
        super().__init__()
        self.conv_layers = []
        for i, f in enumerate(layers2D):
            if f != 'pool':
                self.conv_layers.append(SpikingConv2D(f, kernel_size=kernel_size, X_n=1000,
                                                     robustness_params=robustness_params, name=f'conv_{i+1}'))
            else:
                self.conv_layers.append('pool')
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layers = [SpikingDense(d, X_n=1000, robustness_params=robustness_params, name=f'dense_{i+1}')
                             for i, d in enumerate(layers1D)]
        self.output_layer = SpikingDense(data.num_of_classes, outputLayer=True, robustness_params=robustness_params, name='dense_out')
        self.optimizer = optimizer

    def call(self, x, training=False):
        ti = x
        for layer in self.conv_layers:
            if layer == 'pool':
                ti = MaxMinPool2D()(ti)
            else:
                ti = layer(ti)
        ti = self.flatten(ti)
        for layer in self.dense_layers:
            ti = layer(ti)
        out = self.output_layer(ti)
        return out
