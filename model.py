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



TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


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



# def VGG16(include_top=True, weights='imagenet',
#           input_tensor=None, input_shape=(224, 224, 3),
#           classes=1000):

#         model = Sequential()
#         weight_decay =  0.0005

#         model.add(Conv2D(64, (3, 3), padding='same',
#                          input_shape=[32,32,3],kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.3))

#         model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())

#         model.add(MaxPooling2D(pool_size=(2, 2)))

#         model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.4))

#         model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())

#         model.add(MaxPooling2D(pool_size=(2, 2)))

#         model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.4))

#         model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.4))

#         model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())

#         model.add(MaxPooling2D(pool_size=(2, 2)))


#         model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.4))

#         model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.4))

#         model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())

#         model.add(MaxPooling2D(pool_size=(2, 2)))


#         model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.4))

#         model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.4))

#         model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())

#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.5))

#         model.add(Flatten())
#         model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())

#         model.add(Dropout(0.5))
#         model.add(Dense(10))
#         model.add(Activation('softmax'))
  


#         model.load_weights('weights/cifar10vgg.h5')

#         return model


class SpikingDense(tf.keras.layers.Layer):
    def __init__(self, units, name, X_n=1, outputLayer=False, robustness_params={}, input_dim=None,
                 kernel_regularizer=None, kernel_initializer=None):
        self.units = units
        self.B_n = (1 + 0.5) * X_n
        self.outputLayer=outputLayer
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.noise=robustness_params['noise']
        self.time_bits=robustness_params['time_bits']
        self.weight_bits =robustness_params['weight_bits'] 
        self.w_min, self.w_max=-1.0, 1.0
        self.alpha = tf.cast(tf.fill((units, ), 1), dtype=tf.float64) 
        self.input_dim=input_dim
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer
        super(SpikingDense, self).__init__(name=name)
    
    def build(self, input_dim):
        # In case this is the first dense layer after Flatten layer.
        if input_dim[-1] is None: input_dim=(None, self.input_dim)
        self.kernel = self.add_weight(shape=(input_dim[-1], self.units), name='kernel', regularizer=self.regularizer, initializer=self.initializer)
        self.D_i = self.add_weight(shape=(self.units), initializer=tf.constant_initializer(0), name='D_i')
        self.built = True
    
    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev=tf.Variable(tf.constant(t_min_prev, dtype=tf.float64), trainable=False, name='t_min_prev')
        self.t_min=tf.Variable(tf.constant(t_min, dtype=tf.float64), trainable=False, name='t_min')
        self.t_max=tf.Variable(tf.constant(t_min+self.B_n, dtype=tf.float64), trainable=False, name='t_max')
        return t_min, t_min+self.B_n
            
    def call(self, tj):
        """
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer. 
        """
        output = call_spiking(tj, self.kernel, self.D_i, self.t_min, self.t_max, noise=self.noise)
        # In case of the output layer a simple integration is applied without spiking. 
        if self.outputLayer:
            # Read out the value of membrane potential at time t_min.
            W_mult_x = tf.matmul(self.t_min-tj, self.kernel)
            self.alpha = self.D_i/(self.t_min-self.t_min_prev)
            output = self.alpha * (self.t_min - self.t_min_prev) + W_mult_x
        return output
    
    
class SpikingConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, name, X_n=1, padding='same', kernel_size=(3,3), robustness_params={},
                 kernel_regularizer=None, kernel_initializer=None):
        self.filters=filters
        self.kernel_size=kernel_size
        self.padding=padding
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer
        self.B_n = (1 + 0.5) * X_n
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.w_min, self.w_max=-1.0, 1.0
        self.time_bits=robustness_params['time_bits']
        self.weight_bits =robustness_params['weight_bits'] 
        self.noise=robustness_params['noise']
        self.alpha = tf.cast(tf.fill((filters, ), 1), dtype=tf.float64)
        super(SpikingConv2D, self).__init__(name=name)
    
    def build(self, input_dim):
        self.kernel = self.add_weight(shape=(self.kernel_size[0], self.kernel_size[1], input_dim[-1], self.filters),
                      name='kernel', regularizer=self.regularizer, initializer=self.initializer)
        # Depending on whether there is fusion with batch normalization layer and its position with respect to ReLU activation function the processing in spiking convolutional layer can be different.
        self.BN=tf.Variable(tf.constant([0]), name='BN', trainable=False)
        self.BN_before_ReLU=tf.Variable(tf.constant([0]), name='BN_before_ReLU', trainable=False)
        # When fusing a batch normalization layer with the next convolutional layer where padding=='same', some of the biases in scaled ReLU network are changed, leading to 9 different values.
        self.D_i = self.add_weight(shape=(9, self.filters), initializer=tf.constant_initializer(0), name='D_i')
        self.built = True
    
    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev=tf.Variable(tf.constant(t_min_prev, dtype=tf.float64), trainable=False, name='t_min_prev')
        self.t_min=tf.Variable(tf.constant(t_min, dtype=tf.float64), trainable=False, name='t_min')
        self.t_max=tf.Variable(tf.constant(t_min+self.B_n, dtype=tf.float64), trainable=False, name='t_max')
        return t_min, t_min+self.B_n

    def call(self, tj):
        """
        Input spiking times tj, output spiking times ti. 
        """
        # Image size in case of padding='same' or padding='valid'.
        padding_size, image_same_size = int(self.padding=='same')*(self.kernel_size[0]//2), tf.shape(tj)[1] 
        image_valid_size = image_same_size - self.kernel_size[0]+1
        # Pad input with t_min value, which is equivalent with 0 in ReLU network.
        tj=tf.pad(tj, tf.constant([[0, 0], [padding_size, padding_size,], [padding_size, padding_size], [0, 0]]), constant_values=self.t_min)
        # Extract image patches of size (kernel_size, kernel_size). call_spiking function will be called for different patches in parallel.  
        tj = tf.image.extract_patches(tj, sizes=[1, self.kernel_size[0], self.kernel_size[1], 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        # We reshape input and weights in order to utilize the same function as for the fully-connected layer.
        W = tf.reshape(self.kernel, (-1, self.filters))
        if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1: 
            # In this case the threshold is the same for whole input image.
            tj = tf.reshape(tj, (-1, tf.shape(W)[0]))
            ti = call_spiking(tj, W, self.D_i[0], self.t_min, self.t_max, noise=self.noise)   
            # Layer output is reshaped back.
            if self.padding=='valid':
                ti = tf.reshape(ti, (-1, image_valid_size, image_valid_size, self.filters))
            else:
                ti = tf.reshape(ti, (-1, image_same_size, image_same_size, self.filters))
        else:
            # In this case there are 9 different thresholds for 9 different image partitions.
            tj_partitioned = [tj[:, 1:-1, 1:-1, :], tj[:, :1, :1, :], tj[:, :1, 1:-1, :], tj[:, :1, -1:, :], tj[:, 1:-1, -1:, :], tj[:, -1:, -1:, :] , tj[:, -1:, 1:-1, :], tj[:, -1:, :1, :], tj[:, 1:-1, :1, :]]
            ti_partitioned=[]
            for i, tj_part in enumerate(tj_partitioned):
                # Iterate over 9 different partitions and call call_spiking with different threshold value.
                tj_part = tf.reshape(tj_part, (-1, tf.shape(W)[0]))
                ti_part = call_spiking(tj_part, W, self.D_i[i], self.t_min, self.t_max, noise=self.noise)
                # Partitions are reshaped back.
                if i==0: ti_part=tf.reshape(ti_part, (-1, image_valid_size, image_valid_size, self.filters))
                if i in [1, 3, 5, 7]: ti_part=tf.reshape(ti_part, (-1, 1, 1, self.filters))
                if i in [2, 6]: ti_part=tf.reshape(ti_part, (-1, 1, image_valid_size, self.filters))
                if i in [4, 8]: ti_part=tf.reshape(ti_part, (-1, image_valid_size, 1, self.filters))
                ti_partitioned.append(ti_part) 
            # Partitions are concatenated to create a complete output.
            if image_valid_size!=0:
                ti_top_row = tf.concat([ti_partitioned[1], ti_partitioned[2], ti_partitioned[3]], axis=2)
                ti_middle = tf.concat([ti_partitioned[8], ti_partitioned[0], ti_partitioned[4]], axis=2)
                ti_bottom_row = tf.concat([ti_partitioned[7], ti_partitioned[6], ti_partitioned[5]], axis=2)
                ti = tf.concat([ti_top_row, ti_middle, ti_bottom_row], axis=1)         
            else:
                ti_top_row = tf.concat([ti_partitioned[1], ti_partitioned[3]], axis=2)
                ti_bottom_row = tf.concat([ti_partitioned[7], ti_partitioned[5]], axis=2)
                ti = tf.concat([ti_top_row, ti_bottom_row], axis=1)   
        return ti


class ModelTmax(tf.keras.Model):
    def __init__(self, **kwargs):
        super(ModelTmax, self).__init__(**kwargs)

    def train_step(self, data):
        x, y_all = data
        with tf.GradientTape() as tape:
            y_pred_all = self(x, training=False) 
            loss = self.compiled_loss(y_all, y_pred_all[0], regularization_losses=self.losses)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        t_min_prev, t_min, k=0.0, 1.0, 0
        for layer in self.layers:
            if 'conv' in layer.name or 'dense' in layer.name: 
                try:
                    t_max=t_min + tf.maximum(tf.cast(layer.t_max-layer.t_min, dtype=tf.float64), 10.0*(layer.t_max-tf.reduce_min(y_pred_all[1][k])))
                except IndexError:
                    t_max=0
                layer.t_min_prev.assign(t_min_prev)
                layer.t_min.assign(t_min)
                layer.t_max.assign(t_max)
                t_min_prev, t_min = t_min, t_max
                if k==len(y_pred_all[1]): break
                k+=1
        self.compiled_metrics.update_state(y_all, y_pred_all[0])
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y_all = data
        y_pred_all = self(x, training=False)  
        self.compiled_loss(y_all, y_pred_all[0], regularization_losses=self.losses)
        self.compiled_metrics.update_state(y_all, y_pred_all[0])
        return {m.name: m.result() for m in self.metrics}

def create_vgg_model_ReLU(layers2D, kernel_size, layers1D, data, BN, dropout=0.3 ,optimizer='adam',
                         kernel_regularizer=None, kernel_initializer='glorot_uniform'):
    # inputs = Input(shape=data.input_shape)
    # i_conv = 0
    # i_bn = 0
    # i_dense = 0
    
    # for k, f in enumerate(layers2D):
    #     if f != 'pool':
    #         i_conv += 1
    #         if k == 0:
    #             x = Conv2D(f, kernel_size, padding='same', activation=None,
    #                       kernel_regularizer=kernel_regularizer,
    #                       kernel_initializer=kernel_initializer,
    #                       name='conv2d_'+str(i_conv))(inputs)
    #         else:
    #             x = Conv2D(f, kernel_size, padding='same', activation=None,
    #                       kernel_regularizer=kernel_regularizer,
    #                       kernel_initializer=kernel_initializer,
    #                       name='conv2d_'+str(i_conv))(x)
            
    #         # Changed order: BN before Activation
    #         if BN:
    #             i_bn += 1
    #             x = BatchNormalization(name='batch_normalization_'+str(i_bn))(x)
            
    #         x = tf.keras.layers.Activation('relu')(x)
    #         x = Dropout(dropout)(x)
    #     else:
    #         x = MaxPooling2D()(x)
    
    # x = Flatten()(x)
    
    # for j, d in enumerate(layers1D):
    #     i_dense += 1
    #     x = Dense(d, activation=None,
    #              kernel_regularizer=kernel_regularizer,
    #              kernel_initializer=kernel_initializer,
    #              name='dense_'+str(i_dense))(x)
        
    #     # Changed order for dense layers too
    #     if BN:
    #         i_bn += 1
    #         x = BatchNormalization(name='batch_normalization_'+str(i_bn))(x)
        
    #     x = tf.keras.layers.Activation('relu')(x)
    #     x = Dropout(dropout)(x)
    
    # i_dense += 1
    # x = Dense(data.num_of_classes, activation='softmax',
    #                kernel_regularizer=kernel_regularizer,
    #                name='dense_' + str(i_dense))(x)
    # outputs = x
    # model = Model(inputs=inputs, outputs=outputs)
    # model.compile(metrics=['accuracy'], 
    #              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # Changed to False
    #              optimizer=optimizer)

    model = Sequential()
# Layer 1: Convolutional
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3),
                    padding='same', activation='relu'))
    # Layer 2: Convolutional
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 3: MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 4: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 5: Convolutional
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 6: MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 7: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 8: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 9: Convolutional
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 10: MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 11: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 12: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 13: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 14: MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 15: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 16: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 17: Convolutional
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
    # Layer 18: MaxPooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 19: Flatten
    model.add(Flatten())
    # Layer 20: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 21: Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    # Layer 22: Softmax Layer
    model.add(Dense(units=2, activation='softmax'))

    # Add Optimizer and check accuracy metrics
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy,
                metrics=['accuracy'])
# Check model summary
    return model






import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import optimizers
import numpy as np
from keras import backend as K
from keras import regularizers

class cifar10vgg:
    def __init__(self,train=False):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('weights/cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    
