import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, InputLayer
from tensorflow.keras.models import Model
import numpy as np
import logging
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
import logging


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

class FusedConv2D(tf.keras.layers.Layer):
    """Conv2D layer with properly fused Batch Normalization"""
    def __init__(self, filters, kernel_size, strides=(1,1), padding='same', activation=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        kernel_shape = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, inputs):
        x = tf.nn.conv2d(
            inputs, 
            self.kernel, 
            strides=[1, *self.strides, 1],
            padding=self.padding.upper()
        )
        x = tf.nn.bias_add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x

def fuse_bn_functional(model):
    """Fuse Conv2D + BatchNorm layers in a Keras Functional model"""

    def fuse_conv_bn(conv_layer, bn_layer):
        """Compute fused Conv2D weights and bias"""
        W = conv_layer.get_weights()[0].astype(np.float32)  # (kh, kw, in_ch, out_ch)
        if len(conv_layer.get_weights()) > 1:
            b = conv_layer.get_weights()[1].astype(np.float32)
        else:
            b = np.zeros(conv_layer.filters, dtype=np.float32)

        gamma = bn_layer.gamma.numpy() if bn_layer.scale else np.ones(conv_layer.filters, dtype=np.float32)
        beta = bn_layer.beta.numpy() if bn_layer.center else np.zeros(conv_layer.filters, dtype=np.float32)
        mean = bn_layer.moving_mean.numpy()
        var = bn_layer.moving_variance.numpy()
        epsilon = bn_layer.epsilon

        scale = gamma / np.sqrt(var + epsilon)
        W_fused = W * scale.reshape((1, 1, 1, -1))
        b_fused = beta + (b - mean) * scale
        return W_fused, b_fused


    layer_outputs = {}
    inputs = Input(shape=model.input_shape[1:])
    layer_outputs[model.layers[0].name] = inputs
    layer_inputs = {} 
    for layer in model.layers[1:]:
        inbound_tensors = []
        inbound_names = []
        seen = set()  # track seen layers

        for node in layer._inbound_nodes:
            for inbound_tensor in node.input_tensors:
                inbound_layer, _, _ = inbound_tensor._keras_history
                inbound_layer_name = inbound_layer.name
                if inbound_layer_name in layer_outputs and inbound_layer_name not in seen:
                    inbound_tensors.append(layer_outputs[inbound_layer_name])
                    inbound_names.append(inbound_layer_name)
                    seen.add(inbound_layer_name)


        if not inbound_tensors:
            raise ValueError(f"No inbound tensors found for layer {layer.name}")

        x_in = inbound_tensors[0] if len(inbound_tensors) == 1 else inbound_tensors

        # Save the input tensor for potential fusion later
        layer_inputs[layer.name] = x_in

        # ---- FUSE Conv2D + BN ----
        if isinstance(layer, tf.keras.layers.BatchNormalization) and len(inbound_names) == 1:
            prev_layer = model.get_layer(inbound_names[0])
            if isinstance(prev_layer, tf.keras.layers.Conv2D):
                # Clone Conv2D config but drop activation
                conv_config = prev_layer.get_config()
                act_name = conv_config.get("activation", None)
                conv_config["activation"] = None

                # Correct input channels
                # Use the output tensor of the previous Conv2D, not just layer_inputs
                prev_output = layer_inputs[prev_layer.name]  # this has correct number of channels
                input_channels = prev_output.shape[-1]
                prev_height = layer_inputs[prev_layer.name].shape[1]
                prev_width = layer_inputs[prev_layer.name].shape[2]
                fused_conv = tf.keras.layers.Conv2D(
                    filters=prev_layer.filters,
                    kernel_size=prev_layer.kernel_size,
                    strides=prev_layer.strides,
                    padding=prev_layer.padding,
                    dilation_rate=prev_layer.dilation_rate,
                    use_bias=True,
                    kernel_initializer='zeros',
                    bias_initializer='zeros',
                    name=prev_layer.name + "_fused",
                    input_shape=(prev_height, prev_width, input_channels)  
                )
                x_fused = fused_conv(prev_output)


                # Fuse weights
                # Clone Conv2D layer config
                W_fused, b_fused = fuse_conv_bn(prev_layer, layer)
                fused_conv.set_weights([W_fused, b_fused])


                # W_fused, b_fused = fuse_conv_bn(prev_layer, layer)
                # fused_conv.set_weights([W_fused, b_fused])

                # Re-apply activation if needed
                if act_name and act_name != "linear":
                    x_fused = Activation(act_name)(x_fused)

                layer_outputs[layer.name] = x_fused
                continue  # skip adding BN separately

        # ---- Default: clone layer ----
        new_layer = layer.__class__.from_config(layer.get_config())
# If x_in is a list/tuple of length 1, unpack it
        if isinstance(x_in, (list, tuple)) and len(x_in) == 1:
            x_in = x_in[0]

        x_out = new_layer(x_in)
        if layer.get_weights():
            new_layer.set_weights(layer.get_weights())
        layer_outputs[layer.name] = x_out

    outputs = layer_outputs[model.layers[-1].name]
    return Model(inputs, outputs)

def verify_model_fusion(original_model, fused_model, test_input=None):
    """Thorough verification of model fusion"""
    if test_input is None:
        test_input = np.random.randn(1, *original_model.input_shape[1:]).astype(np.float32)
    
    # Compare outputs
    original_output = original_model.predict(test_input)
    fused_output = fused_model.predict(test_input)
    max_diff = np.abs(original_output - fused_output).max()
    
    # Compare layer-by-layer outputs where possible
    layer_diffs = {}
    for orig_layer in original_model.layers:
        if orig_layer.name + '_fused' in [l.name for l in fused_model.layers]:
            try:
                orig_layer_model = Model(inputs=original_model.input, 
                                        outputs=orig_layer.output)
                fused_layer_model = Model(inputs=fused_model.input, 
                                         outputs=fused_model.get_layer(orig_layer.name + '_fused').output)
                
                orig_out = orig_layer_model.predict(test_input)
                fused_out = fused_layer_model.predict(test_input)
                
                if orig_out.shape == fused_out.shape:
                    layer_diffs[orig_layer.name] = np.abs(orig_out - fused_out).max()
                else:
                    logging.warning(f"Skipping {orig_layer.name} due to shape mismatch: {orig_out.shape} vs {fused_out.shape}")
            except Exception as e:
                logging.warning(f"Couldn't compare {orig_layer.name}: {str(e)}")
    
    return max_diff, layer_diffs
