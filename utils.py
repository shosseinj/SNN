import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, InputLayer
from tensorflow.keras.models import Model
import numpy as np
import logging
import os
import sys
import numpy as np
import tensorflow as tf

def set_up_logging(logging_dir, model_name):
    """
    Set up logging for the simulation.
    """
    os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
           logging.FileHandler(logging_dir + f'/{model_name}_log.txt', mode='w'),
           logging.StreamHandler(sys.stdout),
        ],
    )
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(logging.WARNING)


def get_optimizer(lr):
    """
    Get optimizer for the training on MNIST/Fashion-MNIST dataset.
    """
    learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=5000,  
        decay_rate=0.9, 
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule) 
    return optimizer

class Conv2DWithBias(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1),activation=None, padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)
        self.use_custom_bias = False

    def build(self, input_shape):
        kernel_shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(9, self.filters),
            initializer="zeros",
            trainable=True
        )
        self.std_bias = self.add_weight(
            name="std_bias",
            shape=(self.filters,),
            initializer="zeros",
            trainable=True
        )
        self.BN = self.add_weight(
            name="BN",
            shape=(1,),
            initializer="zeros",
            trainable=False
        )
        self.BN_before_ReLU = self.add_weight(
            name="BN_before_ReLU",
            shape=(1,),
            initializer="zeros",
            trainable=False
        )
        super().build(input_shape)

    def call(self, inputs):
        x = tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding.upper())
        if self.use_custom_bias:
            batch_size = tf.shape(x)[0]
            h, w = tf.shape(x)[1], tf.shape(x)[2]
            h_step = h // 3
            w_step = w // 3

            output = tf.identity(x)  # safe copy

            for i in range(3):
                for j in range(3):
                    region_idx = i * 3 + j
                    region_bias = self.bias[region_idx]  # shape (filters,)

                    h_start, h_end = i * h_step, (i + 1) * h_step
                    w_start, w_end = j * w_step, (j + 1) * w_step

                    output_slice = output[:, h_start:h_end, w_start:w_end, :]
                    output_slice += region_bias  # broadcasting

                    mesh = tf.meshgrid(
                        tf.range(batch_size),
                        tf.range(h_start, h_end),
                        tf.range(w_start, w_end),
                        indexing='ij'
                    )
                    indices = tf.reshape(tf.stack(mesh, axis=-1), [-1, 3])
                    updates = tf.reshape(output_slice, [-1, self.filters])

                    output = tf.tensor_scatter_nd_update(output, indices, updates)

            x = output + self.std_bias
        if self.activation is not None:
            x = self.activation(x)
        return x
    def set_bias(self, bias, W=None, b_term=None):
        self.use_custom_bias = True

        bias = tf.convert_to_tensor(bias, dtype=tf.float32)
        bias = tf.reshape(bias, (-1,))  # shape (filters,)

        if W is not None:
            W = tf.convert_to_tensor(W, dtype=tf.float32)
            if b_term is None:
                b_term = tf.zeros((W.shape[3],), dtype=tf.float32)  # note axis 3 = filters
            else:
                b_term = tf.convert_to_tensor(b_term[:W.shape[3]], dtype=tf.float32)

            b_term = tf.reshape(b_term, (-1,))  # shape (filters,)

            for i in range(9):
                if i == 0:
                    delta_sum_W = tf.zeros((W.shape[3],), dtype=tf.float32)
                elif i == 1:
                    delta_sum_W = tf.reduce_sum(W, axis=(0, 1, 2))  # sum height, width, input channels
                else:
                    delta_sum_W = tf.zeros((W.shape[3],), dtype=tf.float32)

                delta_sum_W = tf.reshape(delta_sum_W, (-1,))  # shape (filters,)
                delta_bias = b_term * delta_sum_W

                # Debug print to verify shapes
                tf.print(f"set_bias i={i} shapes:", 
                        "bias:", tf.shape(bias), bias,
                        "delta_bias:", tf.shape(delta_bias), delta_bias)

                if bias.shape != delta_bias.shape:
                    raise ValueError(f"Shape mismatch: bias {bias.shape} vs delta_bias {delta_bias.shape}")

                adjusted_bias = bias - delta_bias
                adjusted_bias = tf.reshape(adjusted_bias, (self.filters,))

                # self.bias[i].assign(adjusted_bias)
                self.bias[i].assign(tf.cast(adjusted_bias, self.bias[i].dtype))

        else:
            for i in range(9):
                self.bias[i].assign(bias)


class MaxMinPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding
        self.sign = None

    def build(self, input_shape):
        self.sign = self.add_weight(
            name='sign',
            shape=(1, 1, 1, input_shape[-1]),
            initializer='ones',
            trainable=False,
            dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs):
        pooled = tf.nn.max_pool2d(
            inputs * self.sign,
            ksize=[1, *self.pool_size, 1],
            strides=[1, *self.strides, 1],
            padding=self.padding.upper()
        )
        return pooled * self.sign


def create_original_model(input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPool2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)

    return Model(inputs, outputs)


def fuse_bn_functional(original_model):
    inputs = Input(shape=original_model.input_shape[1:])
    x = inputs
    i = 0
    while i < len(original_model.layers):
        layer = original_model.layers[i]

        if isinstance(layer, InputLayer):
            i += 1
            continue

        if isinstance(layer, Conv2D):
            W_and_b = layer.get_weights()
            W = W_and_b[0]
            b = W_and_b[1] if len(W_and_b) > 1 else np.zeros((layer.filters,), dtype=np.float32)

            if (i + 1) < len(original_model.layers) and isinstance(original_model.layers[i + 1], BatchNormalization):
                bn_layer = original_model.layers[i + 1]
                gamma = bn_layer.gamma.numpy()
                beta = bn_layer.beta.numpy()
                moving_mean = bn_layer.moving_mean.numpy()
                moving_var = bn_layer.moving_variance.numpy()
                epsilon = bn_layer.epsilon

                kappa = gamma / np.sqrt(moving_var + epsilon)
                b_fused = beta - moving_mean * kappa
                W_fused = W * kappa

                fused_conv = Conv2DWithBias(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    activation=layer.activation,
                    name=layer.name + '_fused'
                )
                fused_conv.build(x.shape)

                # Weights order:
                # kernel, bias(9, filters), std_bias(filters), BN(1,), BN_before_ReLU(1,)
                weights = [
                    W_fused,
                    np.zeros((9, layer.filters), dtype=np.float32),  # bias placeholder
                    np.zeros((layer.filters,), dtype=np.float32),   # std_bias placeholder
                    np.zeros((1,), dtype=np.float32),                # BN
                    np.zeros((1,), dtype=np.float32)                 # BN_before_ReLU
                ]
                fused_conv.set_weights(weights)
                fused_conv.set_bias(b_fused, W=W_fused)

                x = fused_conv(x)
                i += 2
            else:
                conv = Conv2DWithBias(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                     activation=layer.activation,
                    name=layer.name + '_fused'
                )
                conv.build(x.shape)

                weights = [
                    W,
                    np.zeros((9, layer.filters), dtype=np.float32),  # bias placeholder
                    b,                                               # std_bias placeholder (using bias here)
                    np.zeros((1,), dtype=np.float32),                # BN
                    np.zeros((1,), dtype=np.float32)                 # BN_before_ReLU
                ]
                conv.set_weights(weights)

                x = conv(x)
                i += 1

        elif isinstance(layer, MaxPool2D):
            pool = MaxMinPool2D(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                name=layer.name + '_fused'
            )
            pool.build(x.shape)
            x = pool(x)
            i += 1

        elif isinstance(layer, Activation):
            x = Activation(layer.activation)(x)
            i += 1

        elif isinstance(layer, Flatten):
            x = Flatten()(x)
            i += 1

        elif isinstance(layer, Dense):
            x = Dense(layer.units, activation=layer.activation)(x)
            i += 1

        elif isinstance(layer, Dropout):
            i += 1

        else:
            i += 1

    return Model(inputs, x)
