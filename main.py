


import os
import time
import argparse
import logging
import datetime
import numpy as np
import pickle as pkl
import tensorflow as tf

from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers

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
    def __init__(self, data_name, logging_dir, flatten=False, ttfs_convert=False, ttfs_noise=0.0):
        self.data_name = data_name
        self.flatten = flatten
        self.ttfs_convert = ttfs_convert
        self.ttfs_noise = ttfs_noise
        
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
            self.x_train = 1.0 - self.x_train  # Invert for time-to-first-spike
            self.x_test = 1.0 - self.x_test
        
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
    parser.add_argument('--model_type', type=str, default='SNN', help='Model type: SNN | ReLU')
    parser.add_argument('--model_name', type=str, default='BN', help='Model name (contains FC2 or VGG)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--training', type=strtobool, default=True, help='Enable training mode')
    parser.add_argument('--testing', type=strtobool, default=False, help='Enable testing mode')
    parser.add_argument('--save', type=strtobool, default=True, help='Save model after training')
    parser.add_argument('--load', type=str, default='False', help='Load pre-trained weights')
    parser.add_argument('--findMax', type=strtobool, default=False, help='Find maximum activations per layer')

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
def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
    """Core spiking computation"""
    threshold = t_max - t_min - D_i
    ti = tf.matmul(tj - t_min, W) + threshold + t_min
    ti = tf.where(ti < t_max, ti, t_max)
    ti = ti + tf.random.normal(tf.shape(ti), stddev=robustness_params.get('noise', 0.0))
    return ti

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
            initializer='glorot_uniform', 
            trainable=True, 
            name='kernel',
            regularizer=tf.keras.regularizers.l2(5e-4)
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
        return output

class SpikingConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), X_n=1, padding='same', dropout_rate=0.2, robustness_params={}, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.B_n = (1 + 0.5) * X_n
        self.robustness_params = robustness_params
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], in_channels, self.filters),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel',
            regularizer=tf.keras.regularizers.l2(5e-4)
        )
        kernel_elems = self.kernel_size[0] * self.kernel_size[1] * in_channels
        self.D_i = self.add_weight(
            shape=(kernel_elems, self.filters),
            initializer='zeros',
            trainable=True,
            name='D_i'
        )
        self.t_min_prev = tf.Variable(0.0, trainable=False)
        self.t_min = tf.Variable(0.0, trainable=False)
        self.t_max = tf.Variable(1.0, trainable=False)

    def call(self, tj, training=None):
        padding_size = self.kernel_size[0] // 2 if self.padding == 'same' else 0
        image_size = tf.shape(tj)[1]
        
        tj = tf.pad(
            tj,
            [[0,0], [padding_size, padding_size], [padding_size, padding_size], [0,0]], 
            constant_values=tf.cast(self.t_min.read_value(), tj.dtype)
        )

        tj_patches = tf.image.extract_patches(
            tj, 
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, 1, 1, 1], 
            rates=[1, 1, 1, 1], 
            padding='VALID'
        )
        
        W = tf.reshape(self.kernel, (-1, self.filters))
        tj_flat = tf.reshape(tj_patches, (-1, tf.shape(W)[0]))
        ti = call_spiking(tj_flat, W, self.D_i[0], self.t_min_prev, self.t_min, self.t_max, self.robustness_params)
        ti = tf.reshape(ti, (-1, image_size, image_size, self.filters))
        return ti

class MaxMinPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
    
    def call(self, x):
        x = tf.nn.max_pool2d(x, ksize=self.pool_size, strides=self.pool_size, padding='SAME')
        return x

class VGG_SNN(tf.keras.Model):
    def __init__(self, layers2D, kernel_size, layers1D, data, optimizer, robustness_params):
        super().__init__()

        # Convolutional layers
        self.conv_1 = SpikingConv2D(64, kernel_size=(3,3), X_n=1000,
                                    robustness_params=robustness_params, name='conv_1')
        self.conv_2 = SpikingConv2D(64, kernel_size=(3,3), X_n=1000,
                                    robustness_params=robustness_params, name='conv_2')
        self.pool_1 = MaxMinPool2D(pool_size=2)

        self.conv_3 = SpikingConv2D(128, kernel_size=(3,3), X_n=1000,
                                    robustness_params=robustness_params, name='conv_3')
        self.conv_4 = SpikingConv2D(128, kernel_size=(3,3), X_n=1000,
                                    robustness_params=robustness_params, name='conv_4')
        self.pool_2 = MaxMinPool2D(pool_size=2)

        self.conv_5 = SpikingConv2D(256, kernel_size=(3,3), X_n=1000,
                                    robustness_params=robustness_params, name='conv_5')
        self.conv_6 = SpikingConv2D(256, kernel_size=(3,3), X_n=1000,
                                    robustness_params=robustness_params, name='conv_6')
        self.conv_7 = SpikingConv2D(256, kernel_size=(3,3), X_n=1000,
                                    robustness_params=robustness_params, name='conv_7')
        self.pool_3 = MaxMinPool2D(pool_size=2)

        self.conv_8 = SpikingConv2D(512, kernel_size=(3,3), X_n=1000,
                                    robustness_params=robustness_params, name='conv_8')
        self.conv_9 = SpikingConv2D(512, kernel_size=(3,3), X_n=1000,
                                    robustness_params=robustness_params, name='conv_9')
        self.conv_10 = SpikingConv2D(512, kernel_size=(3,3), X_n=1000,
                                     robustness_params=robustness_params, name='conv_10')
        self.pool_4 = MaxMinPool2D(pool_size=2)

        self.conv_11 = SpikingConv2D(512, kernel_size=(3,3), X_n=1000,
                                     robustness_params=robustness_params, name='conv_11')
        self.conv_12 = SpikingConv2D(512, kernel_size=(3,3), X_n=1000,
                                     robustness_params=robustness_params, name='conv_12')
        self.conv_13 = SpikingConv2D(512, kernel_size=(3,3), X_n=1000,
                                     robustness_params=robustness_params, name='conv_13')
        self.pool_5 = MaxMinPool2D(pool_size=2)

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # Dense layers
        self.dense_1 = SpikingDense(512, X_n=1000, robustness_params=robustness_params, name='dense_1')
        self.dense_out = SpikingDense(
            data.num_of_classes,
            outputLayer=True,
            robustness_params=robustness_params,
            name='dense_out'
        )

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

    def call(self, x, training=False):
        # Forward pass through conv layers
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.pool_1(x)

        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.pool_2(x)

        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        x = self.pool_3(x)

        x = self.conv_8(x)
        x = self.conv_9(x)
        x = self.conv_10(x)
        x = self.pool_4(x)

        x = self.conv_11(x)
        x = self.conv_12(x)
        x = self.conv_13(x)
        x = self.pool_5(x)

        # Flatten
        x = self.flatten(x)

        # Forward pass through dense layers
        x = self.dense_1(x)
        out = self.dense_out(x)

        return out

class SimpleSNN(tf.keras.Model):
    """Simple SNN model for non-VGG architectures"""
    def __init__(self, input_shape, num_classes, robustness_params={}):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = SpikingDense(128, X_n=1000, robustness_params=robustness_params, name='dense_1')
        self.dense2 = SpikingDense(64, X_n=1000, robustness_params=robustness_params, name='dense_2')
        self.output_layer = SpikingDense(
            num_classes, 
            outputLayer=True, 
            robustness_params=robustness_params, 
            name='dense_out'
        )

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
class SpikeMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, x_sample):
        super().__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.x_sample = x_sample  # a small batch to monitor

    def on_epoch_end(self, epoch, logs=None):
        ti = self.x_sample
        # Go through layers
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
        layers2D = [64, 64, 'pool', 128, 128, 'pool',
                    256, 256, 256, 'pool', 512, 512, 512, 'pool',
                    512, 512, 512, 'pool']
        layers1D = [512]
        kernel_size = (3, 3)
        
        model = VGG_SNN(layers2D, kernel_size, layers1D, data, optimizer, robustness_params)
# Initialize SNN timing parameters correctly
        dummy_input = tf.random.normal((1,) + data.input_shape)
        _ = model(dummy_input)
        for layer in model.conv_layers + model.dense_layers + [model.output_layer]:
            if isinstance(layer, SpikingConv2D) or isinstance(layer, SpikingDense):
                layer.t_min.assign(0.0)
                layer.t_max.assign(1.0)
                layer.t_min_prev.assign(0.0)
                layer.D_i.assign(tf.zeros_like(layer.D_i))

        # Try to load and transfer weights from ANN if available
        weights_path = "cifar10vgg.h5"
        if os.path.exists(weights_path):
            try:
            # Load ANN (ReLU) VGG16
                model_ann = VGG16(input_shape=data.input_shape, classes=data.num_of_classes, weights_path=weights_path)
                print("[INFO] ANN weights loaded successfully")

                # Build SNN model with dummy input
                dummy_input = tf.random.normal((1,) + data.input_shape)
                model(dummy_input)  # necessary to initialize weights

                # Transfer Conv2D weights
                ann_conv_layers = [l for l in model_ann.layers if isinstance(l, tf.keras.layers.Conv2D)]
                snn_conv_layers = [l for l in model.conv_layers if isinstance(l, SpikingConv2D)]
                for ann_l, snn_l in zip(ann_conv_layers, snn_conv_layers):
                    snn_l.kernel.assign(ann_l.kernel)
                    print(f"[INFO] Transferred Conv weights: {ann_l.name} → {snn_l.name}")

                # Transfer Dense weights
                ann_dense_layers = [l for l in model_ann.layers if isinstance(l, tf.keras.layers.Dense)]
                snn_dense_layers = model.dense_layers + [model.output_layer]
                for ann_l, snn_l in zip(ann_dense_layers, snn_dense_layers):
                    snn_l.kernel.assign(ann_l.kernel)
                    print(f"[INFO] Transferred Dense weights: {ann_l.name} → {snn_l.name}")

                print("[INFO] ANN → SNN weight transfer complete")

            except Exception as e:
                print(f"[WARNING] Could not transfer ANN weights: {e}")
        else:
            print("[INFO] No pretrained ANN weights found, training from scratch")
                    
        return model
    
    elif args.model_type == 'ReLU':
        # Create ANN model
        if 'VGG' in args.model_name:
            model = VGG16(input_shape=data.input_shape, classes=data.num_of_classes)
        else:
            model = create_simple_relu_model(data.input_shape, data.num_of_classes)
        return model
    
    else:  # SNN but not VGG
        model = SimpleSNN(data.input_shape, data.num_of_classes, robustness_params)
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

    # Dataset
    data = Dataset(
        args.data_name,
        args.logging_dir,
        flatten='FC' in args.model_name,
        ttfs_convert=args.model_type == 'SNN',  # Convert for SNN models
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

    # Model creation
    logging.info("#### Creating the model ####")
    model = create_model(args, data, optimizer, robustness_params)
    
    # Build model
    if hasattr(data, 'input_shape'):
        model.build(input_shape=(None,) + data.input_shape)
    else:
        # For flattened data
        model.build(input_shape=(None, data.x_train.shape[1]))
    
    model.summary()

    # Enable eager execution for debugging if needed
    if args.training and ('SNN' in args.model_type):
        tf.config.run_functions_eagerly(True)

    # Training
    if args.training:
        logging.info("#### Training ####")

        # Compile model
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

        # Test with small dataset first
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

        # Full training
        logging.info("Starting full training...")
        # Pick a small batch of inputs for monitoring
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

    # Testing only mode
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
