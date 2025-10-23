


import os
import time
import argparse
import logging
import datetime
import numpy as np
import pickle as pkl
import tensorflow as tf
from apps import *
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import logging
import numpy as np 
import numpy  
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
# ==============================
# GPU CONFIGURATION
# ==============================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce TF verbosity
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set GPU (use '-1' for CPU only)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.keras.backend.set_floatx('float32')

# ==============================
# UTILITY FUNCTIONS
# ==============================
def set_up_logging(log_dir, model_name):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'{model_name}.log')),
            logging.StreamHandler()
        ]
    )

def get_optimizer(lr):
    """Get optimizer with specified learning rate"""
    return Adam(learning_rate=lr)

# ==============================
# DATASET CLASS
# ==============================
class Dataset:
    def __init__(self, data_name, logging_dir, flatten=False, ttfs_convert=False, ttfs_noise=0.0, T_max=1):
        self.data_name = data_name
        self.flatten = flatten
        self.ttfs_convert = ttfs_convert
        self.ttfs_noise = ttfs_noise
        self.T_max = T_max 
        self.T_start = 1 
        self.T_end = 25 
        # Load dataset
        if data_name == 'MNIST':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
            self.num_of_classes = 10
            self.input_shape = (28, 28, 1)
        elif data_name == 'CIFAR10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
            self.num_of_classes = 10
            self.input_shape = (32, 32, 3)
        elif data_name == 'CIFAR100':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()
            self.num_of_classes = 100
            self.input_shape = (32, 32, 3)
        else:
            raise ValueError(f"Unknown dataset: {data_name}")
        
        # Preprocess data
        self.preprocess_data()
    
    def preprocess_data(self):
        """Preprocess the dataset"""
        # Normalize to [0, 1]
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        # Convert labels to one-hot encoding
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_of_classes)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_of_classes)
        
        # Reshape if needed
        if len(self.x_train.shape) == 3:  # MNIST case (28, 28) -> (28, 28, 1)
            self.x_train = np.expand_dims(self.x_train, -1)
            self.x_test = np.expand_dims(self.x_test, -1)
        
        # TTFS conversion if needed
        if self.ttfs_convert:
            # self.x_train = (1.0 - self.x_train) * self.T_max  # Invert for time-to-first-spike
            # self.x_test = (1.0 - self.x_test) * self.T_max
            self.x_train =   self.T_end-(self.x_train *(self.T_end- self.T_start))               
            self.x_test =   self.T_end-(self.x_test *(self.T_end- self.T_start))
        
        # Add noise if specified
        if self.ttfs_noise > 0:
            self.x_train += np.random.normal(0, self.ttfs_noise, self.x_train.shape)
            self.x_test += np.random.normal(0, self.ttfs_noise, self.x_test.shape)
            # Clip to valid range
            self.x_train = np.clip(self.x_train, 0, 1)
            self.x_test = np.clip(self.x_test, 0, 1)
        
        # Flatten if needed for FC models
        if self.flatten:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)

# ==============================
# CUSTOM CALLBACK
# ==============================
class SaveWeightsEveryNEpochs(Callback):
    def __init__(self, save_path, n=10):
        super().__init__()
        self.save_path = save_path
        self.n = n
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:
            filename = os.path.join(self.save_path, f'weights_epoch_{epoch + 1}.h5')
            # self.model.save_weights(filename)
            print(f"\n[INFO] Saved weights at epoch {epoch + 1} → {filename}")

# ==============================
# ARGUMENT PARSER
# ==============================
def parse_arguments():
    strtobool = lambda s: s.lower() in ['true', '1', 'yes']
    parser = argparse.ArgumentParser(description="SNN/ReLU Training Script")

    parser.add_argument('--data_name', type=str, default='MNIST', help='Dataset: MNIST | CIFAR10 | CIFAR100')
    parser.add_argument('--logging_dir', type=str, default='./logs/', help='Directory for logging')
    parser.add_argument('--weight_dir', type=str, default='./weights/', help='Directory for logging')
    parser.add_argument('--model_type', type=str, default='SNN', help='Model type: SNN | ReLU')
    parser.add_argument('--model_name', type=str, default='BN', help='Model name (contains FC2 or VGG)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--training', type=strtobool, default=True, help='Enable training mode')
    parser.add_argument('--plotDummy', type=strtobool, default=False, help='Enable training mode')
    parser.add_argument('--SNNSummary', type=strtobool, default=True, help='Enable training mode')
    parser.add_argument('--testBeforeTraining', type=strtobool, default=False, help='Enable training mode')
    parser.add_argument('--eagerExcecution', type=strtobool, default=False, help='Enable training mode')
    parser.add_argument('--testing', type=strtobool, default=False, help='Enable testing mode')
    parser.add_argument('--flatten', type=strtobool, default=False, help='Enable testing mode')
    parser.add_argument('--ttfsConvertDataset', type=strtobool, default=True, help='Enable testing mode')
    parser.add_argument('--save', type=strtobool, default=True, help='Save model after training')
    parser.add_argument('--load', type=str, default=True, help='Load pre-trained weights')
    parser.add_argument('--findMax', type=strtobool, default=False, help='Find maximum activations per layer')
    parser.add_argument('--plotExample', type=strtobool, default=False, help='Find maximum activations per layer')

    # Robustness parameters
    parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
    parser.add_argument('--time_bits', type=int, default=0, help='Quantization bits for time')
    parser.add_argument('--weight_bits', type=int, default=0, help='Quantization bits for weights')
    parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Quantile for latency')

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARNING] Ignored args: {unknown}")
    return args

# ==============================
# ANN MODEL FUNCTIONS
# ==============================
def create_simple_relu_model(input_shape, num_classes):
    """Create a simple fully connected ReLU model"""
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def VGG16(input_shape=(32, 32, 3), classes=10, weights_path=None):
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
    
    # Load weights if path provided
    if weights_path and os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"[INFO] Loaded weights from {weights_path}")
        except:
            print(f"[WARNING] Could not load weights from {weights_path}")

    return model

# ==============================
# SNN LAYERS
# ==============================
class MaxMinPool2D(tf.keras.layers.MaxPool2D):
    """
    Max Pooling or Min Pooling operation, depends on the sign of the batch normalization layer before.
    """
    def build(self, input_shape):
        super().build(input_shape)
        # By default the sign is set to 1, which yields max pooling functionality.
        # The sign variable can be changed for some channels when batch normalization is fused with the next convolutonal layer and it changes the sign of the weights. 
        self.sign=tf.Variable(tf.constant(np.ones((1, 1, 1, input_shape[-1]), dtype=np.float32))
, dtype=tf.float32, name='sign', trainable=False)
    def call(self, inputs):
        # Max pooling functionality is called on (self.sign*inputs) input. 
        return super().call(self.sign*inputs)*self.sign



# def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
#     """
#     Calculates spiking times from which ReLU functionality can be recovered.
#     Assumes tau_c=1 and B_i^(n)=1
#     """
#     if robustness_params['time_bits'] != 0:
#         tj = t_min_prev+tf.quantization.fake_quant_with_min_max_args(tf.cast(tj-t_min_prev, dtype=tf.float32),
#             min=t_min_prev, max=t_min, num_bits=robustness_params['time_bits'])
#         tj = tf.cast(tj, tf.float64)
#     if robustness_params['weight_bits'] != 0:
#         W = tf.quantization.fake_quant_with_min_max_args(tf.cast(W, dtype=tf.float32),
#             min=robustness_params['w_min'], max=robustness_params['w_max'], num_bits=robustness_params['weight_bits'])
#         W = tf.cast(W, tf.float64)

#     # Calculate the spiking threshold (Eq. 18)
#     threshold = t_max - t_min - D_i
#     # Calculate output spiking time ti (Eq. 7)
#     ti = (tf.matmul(tj-t_min, W) + threshold + t_min)
#     # Ensure valid spiking time. Do not spike for ti >= t_max.
#     # No spike is modelled as t_max that cancels out in the next layer (tj-t_min) as t_min there is t_max
#     ti = tf.where(ti < t_max, ti, t_max)
#     # Add noise to the spiking time for noise simulations
#     ti = ti + tf.random.normal(tf.shape(ti), stddev=robustness_params['noise'], dtype=ti.dtype)
#     return ti

# def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params, layer_index):
#     """
#     Calculates spiking times from which ReLU functionality can be recovered.
#     FIXED: Preserves variability across layers
#     """
#     # Calculate the weighted input sum
#     weighted_input = tf.matmul(tj - t_min, W)
    
#     # Calculate the range of possible spike times
#     time_range = t_max - t_min
    
#     # FIX: Use gentle normalization instead of aggressive max normalization
#     # Calculate robust statistics across the entire weighted input
#     mean_input = tf.reduce_mean(weighted_input, axis=0, keepdims=True)
#     std_input = tf.math.reduce_std(weighted_input, axis=0, keepdims=True)
    
#     # Avoid division by zero
#     if (std_input.numpy() < 1e-8).any():
#         print("less", layer_index)

#     std_input = tf.maximum(std_input, 1e-8)
    
#     # FIX: Use z-score normalization with controlled scaling
#     # This preserves relative differences while preventing explosion
#     normalized_input = (weighted_input - mean_input) / std_input
    
#     # FIX: Scale to use only a portion of the time range to prevent saturation
#     # Use hyperbolic tangent to smoothly map to the time range
#     scaled_input = tf.tanh(normalized_input / 3.0)  # Division by 3 keeps most values in reasonable range
    
#     # FIX: Map to spike times with better distribution
#     # Center in the time range and subtract thresholds
#     ti = t_min + (time_range / 2.0) * (1.0 - scaled_input) - D_i
    
#     # Alternative simpler approach if above is too complex:
#     # ti = t_max - tf.nn.softplus(weighted_input / std_input) * (time_range / 4.0) - D_i
    
#     # Ensure valid spiking time in [t_min, t_max]
#     ti = tf.clip_by_value(ti, t_min, t_max)
    
#     return ti

def call_spiking(tj_flat, W, D_i_slice, t_min_prev, t_min, t_max, robustness_params=None, layer_index=None):
    # time → strength (soft exponential)
    beta = 3.0  # smaller = more forgiving
    s = tf.exp(-beta * (tj_flat - t_min))

    # ensure no neuron dies, normalize
    s = s / (tf.reduce_sum(s, axis=-1, keepdims=True) + 1e-6)

    # membrane potential
    v = tf.matmul(s, W)

    # SOFT TIME MAPPING — ALWAYS RETURNS VALID SPIKE
    v_norm = tf.sigmoid(v)  # (0,1)
    ti = t_min + (1 - v_norm) * (t_max - t_min)

    return tf.clip_by_value(ti, t_min, t_max)


class SpikingDense(tf.keras.layers.Layer):
    def __init__(self, units, name, outputLayer=False, robustness_params={}, input_dim=None,
                 kernel_regularizer=None, kernel_initializer=None):
        self.units = units

        self.outputLayer=outputLayer
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.robustness_params=robustness_params
        # self.alpha = tf.cast(tf.fill((units, ), 1), dtype=tf.float32) 
        self.alpha = tf.cast(tf.ones((self.units,), dtype=tf.float32), tf.float32)

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
    
    def set_params(self, t_min_prev, t_min,t_max):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev=tf.Variable(tf.constant(t_min_prev, dtype=tf.float32), trainable=False, name='t_min_prev')
        self.t_min=tf.Variable(tf.constant(t_min, dtype=tf.float32), trainable=False, name='t_min')
        self.t_max=tf.Variable(tf.constant(t_max, dtype=tf.float32), trainable=False, name='t_max')
       
            
    def call(self, tj, layer_index):
        """
        Input spiking times tj, output spiking times ti or the value of membrane potential in case of output layer. 
        """
        output = call_spiking(tj, self.kernel, self.D_i, self.t_min_prev, self.t_min, self.t_max, self.robustness_params , layer_index)
        # In case of the output layer a simple integration is applied without spiking. 
        if self.outputLayer:
            # Read out the value of membrane potential at time t_min.
            W_mult_x = tf.matmul(self.t_min-tj, self.kernel)
            self.alpha = self.D_i/(self.t_min-self.t_min_prev)
            output = self.alpha * (self.t_min - self.t_min_prev) + W_mult_x

           
        return output
    
    
class SpikingConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, name, padding='same', kernel_size=(3,3), robustness_params={},
                 kernel_regularizer=None, kernel_initializer=None):
        self.filters=filters
        self.kernel_size=kernel_size
        self.padding=padding
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer
 
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.robustness_params=robustness_params
        # self.alpha = tf.cast(tf.fill((filters, ), 1), dtype=tf.float32)
        self.alpha = tf.cast(tf.ones((filters,), dtype=tf.float32), tf.float32)

        super(SpikingConv2D, self).__init__(name=name)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters),
                      name='kernel', regularizer=self.regularizer, initializer=self.initializer)

        # Depending on whether there is fusion with batch normalization layer and its position with respect to ReLU activation function the processing in spiking convolutional layer can be different.
        self.BN=tf.Variable(tf.constant([0]), name='BN', trainable=False)
        self.BN_before_ReLU=tf.Variable(tf.constant([0]), name='BN_before_ReLU', trainable=False)
        # When fusing a batch normalization layer with the next convolutional layer where padding=='same', some of the biases in scaled ReLU network are changed, leading to 9 different values.
        self.D_i = self.add_weight(shape=(9, self.filters), initializer=tf.constant_initializer(0), name='D_i')
        # self.built = True
    
    def set_params(self, t_min_prev, t_min, t_max):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev=tf.Variable(tf.constant(t_min_prev, dtype=tf.float32), trainable=False, name='t_min_prev')
        self.t_min=tf.Variable(tf.constant(t_min, dtype=tf.float32), trainable=False, name='t_min')
        self.t_max=tf.Variable(tf.constant(t_max, dtype=tf.float32), trainable=False, name='t_max')
     

    def call(self, tj, layer_index):
        """
        Input spiking times tj, output spiking times ti. 
        """
        # print(f"Layer {layer_index}: (tj min={tf.reduce_min(tj).numpy()}   - t_min={self.t_min.numpy()}),            (tj max={tf.reduce_max(tj).numpy()} - t_max={self.t_max.numpy()})")

        # Image size in case of padding='same' or padding='valid'.
        padding_size, image_same_size = int(self.padding=='same')*(self.kernel_size[0]//2), tf.shape(tj)[1] 
        image_valid_size = image_same_size - self.kernel_size[0]+1
        # Pad input with t_min value, which is equivalent with 0 in ReLU network.
        # tj=tf.pad(tj, tf.constant([[0, 0], [padding_size, padding_size,], [padding_size, padding_size], [0, 0]]), constant_values=self.t_min)
        epsilon = tf.random.uniform((), minval=1e-4, maxval=5e-3, dtype=tj.dtype)
        tj = tf.pad(
            tj,
            tf.constant([[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]]),
            constant_values=float(self.t_min + epsilon)
        )

            

        # Extract image patches of size (kernel_size, kernel_size). call_spiking function will be called for different patches in parallel.  
        tj = tf.image.extract_patches(tj, sizes=[1, self.kernel_size[0], self.kernel_size[1], 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        # We reshape input and weights in order to utilize the same function as for the fully-connected layer.
        W = tf.reshape(self.kernel, (-1, self.filters))
        if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1: 
            # In this case the threshold is the same for whole input image.
            tj = tf.reshape(tj, (-1, tf.shape(W)[0]))
            ti = call_spiking(tj, W, self.D_i[0], self.t_min_prev, self.t_min, self.t_max, self.robustness_params, layer_index)
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
                ti_part = call_spiking(tj_part, W, self.D_i[i], self.t_min_prev, self.t_min, self.t_max, self.robustness_params)
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

import matplotlib.pyplot as plt
import numpy as np
def plot_all_spike_histograms(layers_data, layer_names):
    """Plot histograms for all layers in a single figure"""
    n_layers = len(layers_data)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_layers > 1 else [axes]
    
    for i, (layer_data, layer_name) in enumerate(zip(layers_data, layer_names)):
        data_np = layer_data.numpy().flatten()
        
        axes[i].hist(data_np, bins=50, alpha=0.7, edgecolor='black', density=True)
        axes[i].set_xlabel('Spike Time')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'{layer_name}\n'
                         f'Min: {np.min(data_np):.3f}, Max: {np.max(data_np):.3f}\n'
                         f'Mean: {np.mean(data_np):.3f}, Std: {np.std(data_np):.3f}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()


class VGG_SNN(tf.keras.Model):
    def __init__(self,X_n,  optimizer, robustness_params):
        super().__init__()

        self.conv_1 = SpikingConv2D(64, kernel_size=(3,3),  robustness_params=robustness_params, name='conv_1')
        self.conv_2 = SpikingConv2D(64, kernel_size=(3,3),  robustness_params=robustness_params, name='conv_2')
        self.pool_1 = MaxMinPool2D(pool_size=2)

        self.conv_3 = SpikingConv2D(128, kernel_size=(3,3), robustness_params=robustness_params, name='conv_3')
        self.conv_4 = SpikingConv2D(128, kernel_size=(3,3), robustness_params=robustness_params, name='conv_4')
        self.pool_2 = MaxMinPool2D(pool_size=2)

        self.conv_5 = SpikingConv2D(256, kernel_size=(3,3), robustness_params=robustness_params, name='conv_5')
        self.conv_6 = SpikingConv2D(256, kernel_size=(3,3), robustness_params=robustness_params, name='conv_6')
        self.conv_7 = SpikingConv2D(256, kernel_size=(3,3), robustness_params=robustness_params, name='conv_7')
        self.pool_3 = MaxMinPool2D(pool_size=2)

        self.conv_8 = SpikingConv2D(512, kernel_size=(3,3), robustness_params=robustness_params, name='conv_8')
        self.conv_9 = SpikingConv2D(512, kernel_size=(3,3), robustness_params=robustness_params, name='conv_9')
        self.conv_10 = SpikingConv2D(512, kernel_size=(3,3),robustness_params=robustness_params, name='conv_10')
        self.pool_4 = MaxMinPool2D(pool_size=2)

        self.conv_11 = SpikingConv2D(512, kernel_size=(3,3),robustness_params=robustness_params, name='conv_11')
        self.conv_12 = SpikingConv2D(512, kernel_size=(3,3),robustness_params=robustness_params, name='conv_12')
        self.conv_13 = SpikingConv2D(512, kernel_size=(3,3),robustness_params=robustness_params, name='conv_13')
        self.pool_5 = MaxMinPool2D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()

        self.dense_1 = SpikingDense(512,  robustness_params=robustness_params, name='dense_1')
        self.dense_out = SpikingDense(10, outputLayer=False, robustness_params=robustness_params,name='dense_out')

        self.optimizer = optimizer
        self.conv_layers = [
            self.conv_1, self.conv_2, self.pool_1,
            self.conv_3, self.conv_4, self.pool_2,
            self.conv_5, self.conv_6, self.conv_7, self.pool_3,
            self.conv_8, self.conv_9, self.conv_10, self.pool_4,
            self.conv_11, self.conv_12, self.conv_13, self.pool_5
        ]

        self.dense_layers = [self.dense_1]
        self.output_layer = self.dense_out

    def call(self, x):

        layer_outputs = []
        layer_names = []

        layer_outputs.append(x)
        layer_names.append("input_net")
        x = self.conv_1(x,1)
        layer_outputs.append(x)
        layer_names.append("conv_1")
        
        x = self.conv_2(x,2)
        layer_outputs.append(x)
        layer_names.append("conv_2")
        
        x = self.pool_1(x)
        layer_outputs.append(x)
        layer_names.append("pool_1")
        
        x = self.conv_3(x , 3)
        layer_outputs.append(x)
        layer_names.append("conv_3")
        
        x = self.conv_4(x , 4)
        layer_outputs.append(x)
        layer_names.append("conv_4")
        
        x = self.pool_2(x)
        layer_outputs.append(x)
        layer_names.append("pool_2")
        
        x = self.conv_5(x , 5)
        layer_outputs.append(x)
        layer_names.append("conv_5")
        
        x = self.conv_6(x , 6)
        layer_outputs.append(x)
        layer_names.append("conv_6")
        
        x = self.conv_7(x , 7)
        layer_outputs.append(x)
        layer_names.append("conv_7")
        
        x = self.pool_3(x)
        layer_outputs.append(x)
        layer_names.append("pool_3")
        
        x = self.conv_8(x , 8)
        layer_outputs.append(x)
        layer_names.append("conv_8")
        
        x = self.conv_9(x, 9)
        layer_outputs.append(x)
        layer_names.append("conv_9")
        
        x = self.conv_10(x , 10)
        layer_outputs.append(x)
        layer_names.append("conv_10")
        
        x = self.pool_4(x)
        layer_outputs.append(x)
        layer_names.append("pool_4")
        
        x = self.conv_11(x , 11)
        layer_outputs.append(x)
        layer_names.append("conv_11")
        
        x = self.conv_12(x , 12)
        layer_outputs.append(x)
        layer_names.append("conv_12")
        
        x = self.conv_13(x , 13)
        layer_outputs.append(x)
        layer_names.append("conv_13")
        
        x = self.pool_5(x)
        layer_outputs.append(x)
        layer_names.append("pool_5")
        
        x = self.flatten(x)
        layer_outputs.append(x)
        layer_names.append("flatten")
        
        x = self.dense_1(x , 14)
        layer_outputs.append(x)
        layer_names.append("dense_1")
        
        out = self.dense_out(x , 15)
        layer_outputs.append(out)
        layer_names.append("dense_out")

        plot_all_spike_histograms(layer_outputs[:4], layer_names)

        return out


class SpikeMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, x_sample):
        super().__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.x_sample = x_sample  

    def on_epoch_end(self, epoch, logs=None):
        ti = self.x_sample
        for layer in self.model.conv_layers:
            ti = layer(ti)
            if isinstance(layer, SpikingConv2D):
                with self.file_writer.as_default():
                    tf.summary.histogram(f"{layer.name}_spikes", ti, step=epoch)
        
        ti = self.model.flatten(ti)
        for layer in self.model.dense_layers:
            ti = layer(ti)
            if isinstance(layer, SpikingDense):
                with self.file_writer.as_default():
                    tf.summary.histogram(f"{layer.name}_spikes", ti, step=epoch)
        
        out = self.model.output_layer(ti)
        with self.file_writer.as_default():
            tf.summary.histogram(f"{self.model.output_layer.name}_spikes", out, step=epoch)

# ==============================
# MODEL CREATION FUNCTION
# ==============================


def create_model(args, data, optimizer, robustness_params):
    """Create appropriate model based on arguments"""
    
    if 'VGG' in args.model_name:
        if args.load:
            X_n=pkl.load(open(args.weight_dir + args.data_name + '_X_n.pkl', 'rb'))
            logging.info("#### Loading X_n ####", X_n)
    
        model = VGG_SNN(X_n,  optimizer, robustness_params)

        x = data.x_train[0]                       # shape: (32, 32, 3)
        dummy_input = tf.expand_dims(x, axis=0)   # (1, 32, 32, 3)

        if args.plotDummy:
            image = dummy_input[0]
            image_np = image.numpy()
            min_val = image_np.min()
            max_val = image_np.max()

            plt.imshow(image)
            plt.axis('off')
            plt.text(0, 0, f"Min: {min_val:.3f}\nMax: {max_val:.3f}", color='white', 
                    fontsize=12, backgroundcolor='black', ha='left', va='top')

            plt.show()

        logging.info("#### Setting SNN intervals ####")
        X_n= [1 ,1, 220.31496, 158.16324, 47.69856, 56.99223, 25.659113, 27.554497, 10.1943445, 4.2734075, 1.0084844, 1e-6 , 0.22237545, 1e-6 , 1e-6 , 0.1392271, 0.19542553]

        X_n = numpy.array([1 ,1, 220.31496, 158.16324, 47.69856, 56.99223, 
                25.659113, 27.554497, 10.1943445, 4.2734075, 
                1.0084844, 1e-6 , 0.22237545, 1e-6 , 1e-6 , 
                0.1392271, 0.19542553])

        # Avoid zero and huge jumps
        EPS = 1e-6
        X_n = numpy.clip(X_n, EPS, numpy.percentile(X_n, 90))  # limit to 90th percentile max
        X_n = X_n / numpy.max(X_n)  # normalize to [0,1]

        t_min_prev = 0.0
        t_min = 0.0
        t_scale = 1.0  # total window length

        import numpy as np

# === raw X_n from your ANN (keep as-is) ===
        raw_Xn = np.array([1, 1, 220.31496, 158.16324, 47.69856, 56.99223,
                        25.659113, 27.554497, 10.1943445, 4.2734075,
                        1.0084844, 1e-6, 0.22237545, 1e-6, 1e-6,
                        0.1392271, 0.19542553], dtype=np.float64)

        # === Hyperparameters for conversion (tweakable) ===
        EPS = 1e-6                     # safe floor for zeros
        clip_percentile = 90.0         # clip X_n to 90th percentile to avoid outliers
        use_log = True                 # compress dynamic range
        total_spike_window = 10.0      # total time budget across layers (recommended default)
        min_frac_per_layer = 0.02      # min fraction of total window a layer receives (prevents zero-width)

        # === 1) sanitize zeros / tiny values ===
        Xn = np.maximum(raw_Xn, EPS)

        # === 2) clip extreme outliers (optional but recommended) ===
        clip_val = np.percentile(Xn, clip_percentile)
        Xn_clipped = np.minimum(Xn, clip_val)

        # === 3) optional log compression to reduce dynamic range ===
        if use_log:
            # log1p yields log(1 + x) - stable for small and large
            Xc = np.log1p(Xn_clipped)
        else:
            Xc = Xn_clipped.astype(np.float64)

        # === 4) normalize to [0, 1] ===
        Xc_min = Xc.min()
        Xc_max = Xc.max()
        # prevent divide by zero
        if Xc_max - Xc_min < 1e-8:
            Xn_norm = np.ones_like(Xc) * 0.5
        else:
            Xn_norm = (Xc - Xc_min) / (Xc_max - Xc_min)

        # === 5) build per-layer width fraction (ensure minimum share) ===
        layer_fracs = Xn_norm * (1.0 - min_frac_per_layer * len(Xn_norm))  # scale
        # It's simpler / safer to map into [min_frac, 1.0*something]
        layer_fracs = Xn_norm * (1.0 - min_frac_per_layer) + min_frac_per_layer

        # === Normalize fractions so they sum to 1 (optional) or leave proportional:
        layer_fracs = layer_fracs / np.sum(layer_fracs)  # ensures total_spike_window is distributed exactly

        # === 6) compute monotonic t_min/t_max across layers ===
        t_min_prev = 0.0
        t_min = 0.0
        t_max = 0.0
        layer_list = [
            model.conv_1, model.conv_2, model.conv_3, model.conv_4,
            model.conv_5, model.conv_6, model.conv_7, model.conv_8,
            model.conv_9, model.conv_10, model.conv_11, model.conv_12,
            model.conv_13
        ]  # adjust if your model has more/less conv layers

        for i, layer in enumerate(layer_list):
            t_min_prev = t_min
            t_min = t_max
            # each layer receives a time slice proportional to layer_fracs[i]
            delta = float(total_spike_window * layer_fracs[i])
            # ensure a minimal positive delta
            if delta <= 0.0:
                delta = EPS
            t_max = t_min + delta
            # set params on layer (they expect floats)
            try:
                layer.set_params(t_min_prev=float(t_min_prev), t_min=float(t_min), t_max=float(t_max))
            except Exception as e:
                print(f"[WARN] could not set_params on layer {getattr(layer, 'name', i)}: {e}")

        # optionally set for dense layers (flatten -> dense_1 -> dense_out)
        t_min_prev = t_min
        t_min = t_max
        # give FC layers a proportion (sum of remaining fractions or small leftover)
        fc_frac = 1.0 * np.sum(layer_fracs[len(layer_list):]) if len(layer_fracs) > len(layer_list) else 0.05
        delta = float(total_spike_window * max(fc_frac, min_frac_per_layer))
        t_max = t_min + delta
        try:
            model.dense_1.set_params(t_min_prev=float(t_min_prev), t_min=float(t_min), t_max=float(t_max))
            # final layer
            t_min_prev = t_min
            t_min = t_max
            delta = float(total_spike_window * min_frac_per_layer)
            t_max = t_min + delta
            model.dense_out.set_params(t_min_prev=float(t_min_prev), t_min=float(t_min), t_max=float(t_max))
        except Exception as e:
            print(f"[WARN] could not set_params on dense layers: {e}")

        # Debug print - quick sanity check
        print("=== Computed spike windows (t_min,t_max) per conv layer ===")
        t_min_tmp = 0.0
        t_max_tmp = 0.0
        for i, frac in enumerate(layer_fracs[:len(layer_list)]):
            old_t_min = t_min_tmp
            t_min_tmp = t_max_tmp
            t_max_tmp = t_min_tmp + total_spike_window * frac
            print(f"layer {i+1:02d}: t_min={old_t_min:.6f}, t_max={t_max_tmp:.6f}, width={t_max_tmp-old_t_min:.6f}")

        # t_min_prev=0.0
        # t_min = 0.0
        # t_max = 1 / X_n[0]  # Starting with first element, e.g., 220.31496

        # # Layer 1
        # model.conv_1.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[1]  # Next layer's max = previous max + 1 / X_n[1]

        # # Layer 2
        # model.conv_2.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[2]

        # # Layer 3
        # model.conv_3.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[3]

        # # Layer 4
        # model.conv_4.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[4]

        # # Layer 5
        # model.conv_5.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[5]

        # # Layer 6
        # model.conv_6.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[6]

        # # Layer 7
        # model.conv_7.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[7]

        # # Layer 8
        # model.conv_8.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[8]

        # # Layer 9
        # model.conv_9.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[9]

        # # Layer 10
        # model.conv_10.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[10]

        # # Layer 11
        # model.conv_11.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[11]

        # # Layer 12
        # model.conv_12.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[12]

        # # Layer 13
        # model.conv_13.set_params(t_min_prev=t_min_prev, t_min=t_min, t_max=t_max)
        # t_min_prev = t_min
        # t_min = t_max
        # t_max = t_min + 1 / X_n[13]  # If you want to continue for layer 14, update accordingly


        if args.SNNSummary:
            _ = model(dummy_input)
            model.summary()

        weights_path = "cifar10vgg.h5"
        if os.path.exists(weights_path):
    
                model_ann = VGG16(input_shape=data.input_shape, classes=10, weights_path=weights_path)
            
                fused_model = fuse_bn(model_ann, BN='BN', p=-3.0, q=3.0, optimizer=optimizer)
                # logging.info(fused_model.summary())
                if args.plotExample:


                    # ✅ Pick a sample from test data
                    idx = 0  # or random.randint(0, len(data.x_test)-1)
                    x_sample = data.x_test[idx]
                    y_true = data.y_test[idx]

                    # ✅ Ensure correct batch shape
                    x_input = np.expand_dims(x_sample, axis=0)

                    # ✅ Get predictions
                    # Predictions
                    pred_ann = model_ann.predict(x_input)
                    pred_fused = fused_model.predict(x_input)

                    # Flatten if needed
                    pred_ann_flat = np.ravel(pred_ann)
                    pred_fused_flat = np.ravel(pred_fused)

                    # Labels
                    label_ann = np.argmax(pred_ann_flat)
                    label_fused = np.argmax(pred_fused_flat)
                    true_label = np.argmax(y_true) if y_true.ndim > 0 else y_true

                    # Plot
                    plt.figure(figsize=(8, 4))
                    plt.subplot(1, 2, 1)
                    plt.imshow(x_sample.squeeze(), cmap='gray' if x_sample.shape[-1] == 1 else None)
                    plt.title(f"Original Image\nTrue: {true_label}")
                    plt.axis("off")

                    plt.subplot(1, 2, 2)
                    plt.bar(range(10), pred_ann_flat, alpha=0.5, label="ANN", width=0.4)
                    plt.bar(np.arange(10)+0.4, pred_fused_flat, alpha=0.5, label="Fused", width=0.4)
                    plt.xticks(range(10))
                    plt.title(f"Predictions\nANN: {label_ann} | Fused: {label_fused}")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                if args.findMax:
                    logging.info('calculating maximum layer output...')
                    layer_num, X_n = 0, []
                    layers_max = []
                    for k, layer in enumerate(fused_model.layers):
                        if 'conv' in layer.name or 'dense' in layer.name:
                            if k!=len(fused_model.layers)-2:
                                # Calculate X_n of the current layer.
                                layers_max.append(tf.reduce_max(tf.nn.relu(layer.output)))

                    extractor = tf.keras.Model(inputs=fused_model.inputs, outputs=layers_max)
                    output = extractor.predict(data.x_train, batch_size=args.batch_size, verbose=1)
                    import numpy as np

                    X_n = list(map(lambda x: np.max(x), output))
                    logging.info('X_n: %s', X_n)
                    pkl.dump(X_n, open(args.weight_dir + args.data_name + '_X_n.pkl', 'wb'))
                    logging.info('saved maximum layer output')
                print("[INFO] ANN weights loaded successfully")












       
                sample = next(iter(data.x_train))  # shape (32, 32, 3)
                sample_batch = tf.expand_dims(sample, axis=0)  # shape (1, 32, 32, 3)
                model(sample_batch)  # necessary to initialize weights

                # Transfer Conv2D weights
                X_n= [220.31496, 158.15593, 47.710045, 56.97334, 25.651484, 27.545849, 10.183872, 4.269333, 1.0077846, 0.0, 0.22237545, 0.0, 0.0, 0.1392271, 0.19542684]
                ann_conv_layers = [l for l in model_ann.layers if isinstance(l, tf.keras.layers.Conv2D)]
                snn_conv_layers = [l for l in model.conv_layers if isinstance(l, SpikingConv2D)]
                for idx, (ann_l, snn_l) in enumerate(zip(ann_conv_layers, snn_conv_layers)):
                    if X_n[idx] > 0:  # avoid division by zero
                        scaled_kernel = ann_l.kernel / X_n[idx]
                    else:
                        scaled_kernel = ann_l.kernel 
                    snn_l.kernel.assign(scaled_kernel)
                    print(f"[INFO] Transferred Conv weights: {ann_l.name} → {snn_l.name}")

                # Transfer Dense weights
                offset = len(ann_conv_layers) 
                ann_dense_layers = [l for l in model_ann.layers if isinstance(l, tf.keras.layers.Dense)]
                snn_dense_layers = model.dense_layers + [model.output_layer]
                for idx,(ann_l, snn_l) in enumerate(zip(ann_dense_layers, snn_dense_layers)):
                    if X_n[offset + idx] > 0:
                        scaled_kernel = ann_l.kernel / X_n[offset + idx]
                    else:
                        scaled_kernel = ann_l.kernel
                    snn_l.kernel.assign(scaled_kernel)                   

                    print("[INFO] ANN → SNN weight transfer complete")

        
        else:
            print("[INFO] No pretrained ANN weights found, training from scratch")
                    
        return model

    


# ==============================
# MAIN TRAINING FUNCTION
# ==============================
def main():
    start_time = time.time()
    args = parse_arguments()
    args.model_name = args.data_name + "-" + args.model_name

    # Setup logging
    set_up_logging(args.logging_dir, args.model_name)
    logging.info("### Starting Training Script ###")

    # Robustness parameters
    robustness_params = {
        'noise': args.noise,
        'time_bits': args.time_bits,
        'weight_bits': args.weight_bits,
        'latency_quantiles': args.latency_quantiles
    }

    data = Dataset(
        args.data_name,
        args.logging_dir,
        flatten= args.flatten,
        ttfs_convert= args.ttfsConvertDataset,
        ttfs_noise=args.noise,
    )

    # Optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.lr,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=1.0)

    logging.info("#### Creating the model ####")
    model = create_model(args, data, optimizer, robustness_params)
    

    if args.eagerExcecution:
        tf.config.run_functions_eagerly(True)

    # Training
    if args.training:
        logging.info("#### Training ####")
        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        # Callbacks
        os.makedirs("weights", exist_ok=True)
        save_cb = SaveWeightsEveryNEpochs("weights/", n=5)
        checkpoint_cb = ModelCheckpoint(
            "weights/best_model",
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True
        )
        
        log_dir = os.path.join("logs", args.model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

        if args.testBeforeTraining:
            logging.info("Testing with small batch...")
            try:
                # Use smaller subset for initial test
                train_subset = min(1000, len(data.x_train))
                test_subset = min(500, len(data.x_test))
                
                history = model.fit(
                    data.x_train[:train_subset],
                    data.y_train[:train_subset],
                    batch_size=min(32, args.batch_size),
                    epochs=1,
                    validation_data=(data.x_test[:test_subset], data.y_test[:test_subset]),
                    verbose=1
                )
                logging.info("Small batch training successful!")
            except Exception as e:
                logging.error(f"Training failed: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return


        logging.info("Starting full training...")

        x_sample = data.x_train[:32]
        spike_monitor_cb = SpikeMonitorCallback(log_dir=os.path.join("logs", args.model_name), x_sample=x_sample)

        history = model.fit(
            data.x_train,
            data.y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(data.x_test, data.y_test),
            callbacks=[tensorboard_cb, save_cb, checkpoint_cb, spike_monitor_cb],
            verbose=1
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(data.x_test, data.y_test, verbose=1)
        logging.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

        if args.save:
            model.save_weights("weights/final_model.h5")
            logging.info("Model saved → weights/final_model.h5")

    elif args.testing:
        logging.info("#### Testing ####")
        if args.load and args.load != 'False':
            try:
                model.load_weights(args.load)
                logging.info(f"Loaded weights from {args.load}")
            except:
                logging.error(f"Could not load weights from {args.load}")
                return
        
        test_loss, test_acc = model.evaluate(data.x_test, data.y_test, verbose=1)
        logging.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    logging.info(f"### Total Elapsed Time: {time.time() - start_time:.2f} sec ###")

if __name__ == "__main__":
    main()
