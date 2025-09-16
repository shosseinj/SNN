# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, InputLayer
# from tensorflow.keras.models import Model
# import numpy as np
# import logging
# import os
# import sys
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Flatten, Dense, Dropout
# from tensorflow.keras.models import Model
# import numpy as np
# import logging


# def set_up_logging(logging_dir, model_name):
#     """
#     Set up logging for the simulation.
#     """
#     os.makedirs(logging_dir, exist_ok=True)
#     logging.basicConfig(
#         level=logging.DEBUG,
#         handlers=[
#            logging.FileHandler(logging_dir + f'/{model_name}_log.txt', mode='w'),
#            logging.StreamHandler(sys.stdout),
#         ],
#     )
#     mpl_logger = logging.getLogger("matplotlib")
#     mpl_logger.setLevel(logging.WARNING)


# def get_optimizer(lr):
#     """
#     Get optimizer for the training on MNIST/Fashion-MNIST dataset.
#     """
#     learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=lr,
#         decay_steps=5000,  
#         decay_rate=0.9, 
#     )
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule) 
#     return optimizer

# class FusedConv2D(tf.keras.layers.Layer):
#     """Conv2D layer with properly fused Batch Normalization"""
#     def __init__(self, filters, kernel_size, strides=(1,1), padding='same', activation=None, **kwargs):
#         super().__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.padding = padding
#         self.activation = tf.keras.activations.get(activation)
        
#     def build(self, input_shape):
#         kernel_shape = self.kernel_size + (input_shape[-1], self.filters)
#         self.kernel = self.add_weight(
#             name='kernel',
#             shape=kernel_shape,
#             initializer='glorot_uniform',
#             trainable=True
#         )
#         self.bias = self.add_weight(
#             name='bias',
#             shape=(self.filters,),
#             initializer='zeros',
#             trainable=True
#         )
#         super().build(input_shape)
        
#     def call(self, inputs):
#         x = tf.nn.conv2d(
#             inputs, 
#             self.kernel, 
#             strides=[1, *self.strides, 1],
#             padding=self.padding.upper()
#         )
#         x = tf.nn.bias_add(x, self.bias)
#         if self.activation is not None:
#             x = self.activation(x)
#         return x

# def fuse_bn_functional(model):
#     """Fuse Conv2D + BatchNorm layers in a Keras Functional model"""

#     def fuse_conv_bn(conv_layer, bn_layer):
#         """Compute fused Conv2D weights and bias"""
#         W = conv_layer.get_weights()[0].astype(np.float32)  # (kh, kw, in_ch, out_ch)
#         if len(conv_layer.get_weights()) > 1:
#             b = conv_layer.get_weights()[1].astype(np.float32)
#         else:
#             b = np.zeros(conv_layer.filters, dtype=np.float32)

#         gamma = bn_layer.gamma.numpy() if bn_layer.scale else np.ones(conv_layer.filters, dtype=np.float32)
#         beta = bn_layer.beta.numpy() if bn_layer.center else np.zeros(conv_layer.filters, dtype=np.float32)
#         mean = bn_layer.moving_mean.numpy()
#         var = bn_layer.moving_variance.numpy()
#         epsilon = bn_layer.epsilon

#         scale = gamma / np.sqrt(var + epsilon)
#         W_fused = W * scale.reshape((1, 1, 1, -1))
#         b_fused = beta + (b - mean) * scale
#         return W_fused, b_fused


#     layer_outputs = {}
#     inputs = Input(shape=model.input_shape[1:])
#     layer_outputs[model.layers[0].name] = inputs
#     layer_inputs = {} 
#     for layer in model.layers[1:]:
#         inbound_tensors = []
#         inbound_names = []
#         seen = set()  # track seen layers

#         for node in layer._inbound_nodes:
#             for inbound_tensor in node.input_tensors:
#                 inbound_layer, _, _ = inbound_tensor._keras_history
#                 inbound_layer_name = inbound_layer.name
#                 if inbound_layer_name in layer_outputs and inbound_layer_name not in seen:
#                     inbound_tensors.append(layer_outputs[inbound_layer_name])
#                     inbound_names.append(inbound_layer_name)
#                     seen.add(inbound_layer_name)


#         if not inbound_tensors:
#             raise ValueError(f"No inbound tensors found for layer {layer.name}")

#         x_in = inbound_tensors[0] if len(inbound_tensors) == 1 else inbound_tensors

#         # Save the input tensor for potential fusion later
#         layer_inputs[layer.name] = x_in

#         # ---- FUSE Conv2D + BN ----
#         if isinstance(layer, tf.keras.layers.BatchNormalization) and len(inbound_names) == 1:
#             prev_layer = model.get_layer(inbound_names[0])
#             if isinstance(prev_layer, tf.keras.layers.Conv2D):
#                 # Clone Conv2D config but drop activation
#                 conv_config = prev_layer.get_config()
#                 act_name = conv_config.get("activation", None)
#                 conv_config["activation"] = None

#                 # Correct input channels
#                 # Use the output tensor of the previous Conv2D, not just layer_inputs
#                 prev_output = layer_inputs[prev_layer.name]  # this has correct number of channels
#                 input_channels = prev_output.shape[-1]
#                 prev_height = layer_inputs[prev_layer.name].shape[1]
#                 prev_width = layer_inputs[prev_layer.name].shape[2]
#                 fused_conv = tf.keras.layers.Conv2D(
#                     filters=prev_layer.filters,
#                     kernel_size=prev_layer.kernel_size,
#                     strides=prev_layer.strides,
#                     padding=prev_layer.padding,
#                     dilation_rate=prev_layer.dilation_rate,
#                     use_bias=True,
#                     kernel_initializer='zeros',
#                     bias_initializer='zeros',
#                     name=prev_layer.name + "_fused",
#                     input_shape=(prev_height, prev_width, input_channels)  
#                 )
#                 x_fused = fused_conv(prev_output)


#                 # Fuse weights
#                 # Clone Conv2D layer config
#                 W_fused, b_fused = fuse_conv_bn(prev_layer, layer)
#                 fused_conv.set_weights([W_fused, b_fused])


#                 # W_fused, b_fused = fuse_conv_bn(prev_layer, layer)
#                 # fused_conv.set_weights([W_fused, b_fused])

#                 # Re-apply activation if needed
#                 if act_name and act_name != "linear":
#                     x_fused = Activation(act_name)(x_fused)

#                 layer_outputs[layer.name] = x_fused
#                 continue  # skip adding BN separately

#         # ---- Default: clone layer ----
#         new_layer = layer.__class__.from_config(layer.get_config())
# # If x_in is a list/tuple of length 1, unpack it
#         if isinstance(x_in, (list, tuple)) and len(x_in) == 1:
#             x_in = x_in[0]

#         x_out = new_layer(x_in)
#         if layer.get_weights():
#             new_layer.set_weights(layer.get_weights())
#         layer_outputs[layer.name] = x_out

#     outputs = layer_outputs[model.layers[-1].name]
#     return Model(inputs, outputs)

# def verify_model_fusion(original_model, fused_model, test_input=None):
#     """Thorough verification of model fusion"""
#     if test_input is None:
#         test_input = np.random.randn(1, *original_model.input_shape[1:]).astype(np.float32)
    
#     # Compare outputs
#     original_output = original_model.predict(test_input)
#     fused_output = fused_model.predict(test_input)
#     max_diff = np.abs(original_output - fused_output).max()
    
#     # Compare layer-by-layer outputs where possible
#     layer_diffs = {}
#     for orig_layer in original_model.layers:
#         if orig_layer.name + '_fused' in [l.name for l in fused_model.layers]:
#             try:
#                 orig_layer_model = Model(inputs=original_model.input, 
#                                         outputs=orig_layer.output)
#                 fused_layer_model = Model(inputs=fused_model.input, 
#                                          outputs=fused_model.get_layer(orig_layer.name + '_fused').output)
                
#                 orig_out = orig_layer_model.predict(test_input)
#                 fused_out = fused_layer_model.predict(test_input)
                
#                 if orig_out.shape == fused_out.shape:
#                     layer_diffs[orig_layer.name] = np.abs(orig_out - fused_out).max()
#                 else:
#                     logging.warning(f"Skipping {orig_layer.name} due to shape mismatch: {orig_out.shape} vs {fused_out.shape}")
#             except Exception as e:
#                 logging.warning(f"Couldn't compare {orig_layer.name}: {str(e)}")
    
#     return max_diff, layer_diffs



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


class Conv2DWithBias(tf.keras.layers.Conv2D):
    """
    Convolutional layer with potentially different bias at different locations.
    """
    def build(self, input_shape):
        super().build(input_shape)
        # These two variables determine the processing of the layer.
        self.BN=tf.Variable(tf.constant([0]), name='BN', trainable=False)
        self.BN_before_ReLU=tf.Variable(tf.constant([0]), name='BN_before_ReLU', trainable=False)
        
    def set_bias(self, bias, W=None, b_term=[0.0]):
        """
        Creates bias variable and changes bias on certain locations when needed.
        """
        # The bias Variable is added in this function and can have 9 potential values for each filter.
        # W can represent the kernel before fusion and b_term corresponds to the term which is multiplied with kernel (see Eqs. 9, 11, etc.).  
        self.bias = self.add_weight(shape=(9, self.filters), initializer='zeros', dtype=tf.float32, name='bias')
        self.use_custom_bias = True
        self.use_bias = False  # disable standard bias logic
        if W is not None: 
            # Calculate the overall kernel summation.
            W_sum_2D  = tf.math.reduce_sum(W, axis=(0, 1))
            b_term = b_term[:tf.shape(W)[2]]
        for i in range(9):
            # delta_sum_W calculates kernel summation of the weights which correspond to the zero-padded inputs for the particular image part. 
            if i==0:
                # This branch corresponds to the inner part of the image which has unchanged bias.
                b_term=tf.cast(b_term, dtype=tf.float32)
                # delta_sum_W is equal to 0 which yields unchanged bias.
                delta_sum_W = tf.zeros((tf.shape(b_term)[0], 1), dtype=tf.float32)
            elif i==1:
                # This branch corresponds to the top left corner of the image, etc. 
                delta_sum_W = (W_sum_2D - tf.reduce_sum(W[1:, 1:, :, :], axis=[0, 1]))
            elif i==2:
                delta_sum_W = tf.reduce_sum(W[:1, :, :, :], axis=[0, 1])
            elif i==3:
                delta_sum_W = (W_sum_2D - tf.reduce_sum(W[1:, :-1, :, :], axis=[0, 1]))
            elif i==4:
                 delta_sum_W = tf.reduce_sum(W[:, -1:, :, :], axis=[0, 1])
            elif i==5:
                delta_sum_W = (W_sum_2D - tf.reduce_sum(W[:-1, :-1, :, :], axis=[0, 1]))
            elif i==6:
                delta_sum_W = tf.reduce_sum(W[-1:, :, :, :], axis=[0, 1])
            elif i==7:
                delta_sum_W = (W_sum_2D - tf.reduce_sum(W[:-1, 1:, :, :], axis=[0, 1]))
            elif i==8:
                delta_sum_W = tf.reduce_sum(W[:, :1, :, :], axis=[0, 1])
            # See Eqs. 11, 14.  
            delta_bias = tf.reduce_sum(tf.matmul(tf.linalg.diag(b_term), delta_sum_W), axis=0)
            # The bias is decreased for the value which corresponds to the terms which come from zero-padding. 
            self.bias[i].assign(bias - delta_bias)
            # When padding=='valid' or there is no batch normalization layer, or the batch normalization layer is fused with the previous convolutional layer, all locations have the same bias and we break from the for loop. 
            if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1: break
                
    def call(self, inputs):
        # Convolution operation is initially called. 
        #result = self._convolution_op(inputs, self.kernel)  #incompatible with never TF
        result = super().call(inputs)
        if self.use_custom_bias:
            if self.padding=='valid' or self.BN!=1 or self.BN_before_ReLU==1:
                #When padding=='valid' or there is no batch normalization layer, or the batch normalization layer is fused with the previous convolutional layer, all locations have the same bias.
                result=result + self.bias[0]
            else:
                #Otherwise we add different bias to 9 different locations.
                result_0 = result[:, 1:-1, 1:-1, :] + self.bias[0]
                result_1 = result[:, :1, :1, :] + self.bias[1]
                result_2 = result[:, :1, 1:-1, :] + self.bias[2]
                result_3 = result[:, :1, -1:, :] + self.bias[3]
                result_4 = result[:, 1:-1, -1:, :] + self.bias[4]
                result_5 = result[:, -1:, -1:, :] + self.bias[5]
                result_6 = result[:, -1:, 1:-1, :] + self.bias[6]
                result_7 = result[:, -1:, :1, :] + self.bias[7]
                result_8 = result[:, 1:-1, :1, :] + self.bias[8]
                # Parts of image are concatenated to generate a complete output.
                top_row = tf.concat([result_1, result_2, result_3], axis=2)
                middle = tf.concat([result_8, result_0, result_4], axis=2)
                bottom_row = tf.concat([result_7, result_6, result_5], axis=2)
                result = tf.concat([top_row, middle, bottom_row], axis=1)
        return result  


class MaxMinPool2D(tf.keras.layers.MaxPool2D):
    """
    Max Pooling or Min Pooling operation, depends on the sign of the batch normalization layer before.
    """
    def build(self, input_shape):
        super().build(input_shape)
        # By default the sign is set to 1, which yields max pooling functionality.
        # The sign variable can be changed for some channels when batch normalization is fused with the next convolutonal layer and it changes the sign of the weights. 
        self.sign = tf.Variable(
            tf.constant(np.ones((1, 1, 1, input_shape[-1]), dtype=np.float32)), 
            dtype=tf.float32, 
            name='sign', 
            trainable=False
        )
    def call(self, inputs):
        # Max pooling functionality is called on (self.sign*inputs) input. 
        return super().call(self.sign*inputs)*self.sign


def copy_layer(orig_layer):
    """
    Deep copy of a layer with MaxPooling2D layer being replaced with MaxMinPool2D and Conv2D with Conv2DWithBias layer.
    """
    config = orig_layer.get_config()
    if 'pool' in orig_layer.name:
        layer = MaxMinPool2D()
    elif 'conv' in orig_layer.name: 
        config['use_bias']=False
        layer = Conv2DWithBias.from_config(config)
    else:
        layer = type(orig_layer).from_config(config)
    layer.build(orig_layer.input_shape)
    return layer


def copy_model(fused_model, model, i):
    """
    Deep copy of a model and exchange Conv with Conv2DWithBias layers and MaxPool with MaxMinPool layers.
    The layers which are not used during inference are dropped.
    """
    while i < len(model.layers):
        while 'dropout' in model.layers[i].name or 'activity_regularization' in model.layers[i].name: i+=1
        # Deep copy of layer.
        fused_layer = copy_layer(model.layers[i])
        if 'conv' in model.layers[i].name:
            W, b = model.layers[i].get_weights()
            # Set parameters of Conv2DWithBias layer.
            fused_layer.set_weights([W, np.array([0]), np.array([0])])
            fused_layer.set_bias(bias=b)
        elif 'dense' in model.layers[i].name:
            # Set parameters of fully-connected layer.
            fused_layer.set_weights(model.layers[i].get_weights())
        fused_model.add(fused_layer)
        i+=1


def fuse_bn(model, p, q, optimizer, BN = True, BN_before_ReLU = False):
    """
    Creates new models which:
        Fuses all (imaginary) batch normalization layers; 
        Changes bias on locations where it is needed; 
        Transforms MaxPooling layers in MaxMinPooling layers and Conv2D layers in Conv2DWithBias.  
    """
    fused_model = tf.keras.Sequential()
    # Add input layer.
    fused_model.add(copy_layer(model.layers[0]))
    i=1
    # If condition is satisfied, there is an imaginary batch normalization layer which is merged.
    if not (p==0 and q==1): i = fuse_imaginary_bn(fused_model, model, p, q)
    if BN:
        # There are batch normalization layers.
        if BN_before_ReLU:
            # Batch normalization layers are always found before ReLU activation function.
            while i<len(model.layers):
                if 'batch_norm' in model.layers[i].name:
                    # Fuse this batch normalization layer with previous convolutional or fully-connected layer. 
                    i = fuse_bn_before_activation(fused_model, model, i)
                if i==(len(model.layers)-1):
                    # Add last Dense layer with 'softmax'.
                    layer = copy_layer(model.layers[-2])
                    layer.set_weights(model.layers[-2].get_weights())
                    fused_model.add(layer)
                    fused_model.add(copy_layer(model.layers[-1]))
                i+=1
        else:
            # Batch normalization layers are always found after ReLU activation function.
            while i<len(model.layers): 
                # Add first Dense or Convolutional layer if there was no imaginary batch normalization.
                if (p==0 and q==1) and i==1:
                    layer = copy_layer(model.layers[1])
                    if 'conv' in model.layers[1].name:
                        kernel, bias = model.layers[1].get_weights()
                        # Set BN flag to 0 and BN_before_ReLU to 0.
                        layer.set_weights([kernel, np.array([0]), np.array([0])])
                        # Bias is same for all locations.
                        layer.set_bias(bias)
                    else:
                        layer.set_weights(model.layers[1].get_weights())
                    fused_model.add(layer)
                    fused_model.add(copy_layer(model.layers[2]))
                if 'batch_norm' in model.layers[i].name:
                    # Fuse this batch normalization layer with next convolutional or fully-connected layer.
                    i = fuse_bn_after_activation(fused_model, model, i)
                i+=1
    else:
        # If there is no batch normalization layers, copy model such that Conv2D and MaxPooling layers are replaced with ConvWithBias and MaxMinPooling respectively. 
        copy_model(fused_model, model, i)
    fused_model.compile(metrics=['accuracy'], loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=optimizer)  
    return fused_model


def fuse_imaginary_bn(fused_model, model, p, q): 
    """
    Fuse an imaginary batch normalization layer due to an input on arbitrary [p, q] range different from [0, 1].
    """
    first_layer = model.layers[1]
    input_image_shape, _, input_channels, _ = tf.shape(first_layer.kernel)
    kappa = tf.cast(tf.fill((input_channels), value=q-p), dtype=tf.float32)
    b_term=tf.cast(tf.fill((input_channels), value=p), dtype=tf.float32)
    if 'conv' in first_layer.name:
        kappa=tf.tile(kappa, [input_image_shape**2])
        b_term=tf.tile(b_term, [input_image_shape**2])
    W = tf.reshape(first_layer.kernel, (-1, first_layer.filters))
    kappa = tf.linalg.diag(kappa)
    W_fused = tf.matmul(kappa, W)
    # See Eq. 13. 
    W_fused = tf.reshape(W_fused, tf.shape(first_layer.kernel))   
    # See Eq. 12. 
    b_fused = first_layer.bias + tf.reduce_sum(tf.matmul(tf.linalg.diag(b_term), W), axis=0)
    # Copy first convolutional or fully-connected layer.
    layer = copy_layer(first_layer)
    if 'conv' in first_layer.name:
        # Set BN flag to 1 and BN_before_ReLU to 0.
        layer.set_weights([W_fused, np.array([1]), np.array([0])])
        # Create bias which will have 9 different values. 
        # Those values are generated by subtracting from b_fused the terms which come through padded input.
        # The obtained results is in Eq. 14. 
        layer.set_bias(bias=b_fused, W=first_layer.kernel, b_term=b_term)
    else:
        layer.set_weights([W_fused, b_fused])
    fused_model.add(layer)
    fused_model.add(copy_layer(model.layers[2]))  
    return 3


def fuse_bn_before_activation(fused_model, model, i):
    """
    Fuses batch normalization layer with previous layer.
    """
    bn = model.layers[i]
    kappa = tf.linalg.diag(bn.gamma/tf.sqrt(bn.epsilon + bn.moving_variance))
    previous_layer = model.layers[i-1]
    output_shape = tf.shape(previous_layer.kernel)[-1]
    W = tf.reshape(previous_layer.kernel, (-1, output_shape))
    # See Eq. 8.
    W_fused = tf.transpose(tf.matmul(kappa, tf.transpose(W)))
    W_fused = tf.reshape(W_fused, tf.shape(previous_layer.kernel))  
    # See Eq. 7.
    b_fused = bn.beta - bn.moving_mean*tf.linalg.diag_part(kappa)
    b_fused += tf.squeeze(tf.matmul(kappa, previous_layer.bias[:, tf.newaxis]))
    layer = copy_layer(previous_layer)
    if 'conv' in previous_layer.name:
        # Set BN flag to 1 and BN_before_ReLU to 1.
        layer.set_weights([W_fused, np.array([1]), np.array([1])])
        # Bias is same everywhere.
        layer.set_bias(b_fused)
    else:
        layer.set_weights([W_fused, b_fused])
    fused_model.add(layer)
    fused_model.add(copy_layer(model.layers[i+1]))
    # Skip Dropout and ActivityRegularization layers. 
    while 'dropout' in model.layers[i+2].name or 'activity_regularization' in model.layers[i+2].name: i+=1
    # Add Flatten layer when it appears. 
    if 'flatten' in model.layers[i+2].name or 'pool' in model.layers[i+2].name: 
        fused_model.add(copy_layer(model.layers[i+2]))
        i+=1
    return i+1

        
def fuse_bn_after_activation(fused_model, model, i): 
    """
    Fuse batch normalization with following layer.
    """
    bn = model.layers[i]
    kappa = bn.gamma/tf.sqrt(bn.epsilon + bn.moving_variance)
    # See Eq. 9. 
    b_term = bn.beta - bn.moving_mean*kappa
    # Skip Dropout and ActivityRegularization layers. 
    while 'dropout' in model.layers[i+1].name: i+=1
    # if there is a MaxPooling layer in before the next parameterized layer, the sign will be changed for the channels where it is needed. 
    if 'max_pool' in model.layers[i+1].name:
        mp = model.layers[i+1]
        mmp = MaxMinPool2D() 
        mmp.build(mp.input_shape)
        # Change sign to the sign of kappa. 
        mmp.sign.assign(tf.math.sign(kappa)[tf.newaxis, tf.newaxis, tf.newaxis, :])
        fused_model.add(mmp)
        i+=1
        if 'dropout' in model.layers[i+1].name: i+=1
    if 'flatten' in model.layers[i+1].name:
        # Add Flatten layer when it appears.
        fused_model.add(copy_layer(model.layers[i+1]))
        ft = model.layers[i+1]
        kappa = tf.tile(kappa, [ft.output_shape[-1]//ft.input_shape[-1]])
        b_term = tf.tile(tf.squeeze(b_term), [ft.output_shape[-1]//ft.input_shape[-1]])
        i+=1
    next_layer = model.layers[i+1]
    input_image_shape = tf.shape(next_layer.kernel)[0]
    output_shape = tf.shape(next_layer.kernel)[-1]
    if 'conv' in next_layer.name:
        kappa=tf.tile(kappa, [input_image_shape**2])
        b_term=tf.tile(b_term, [input_image_shape**2])
    W = tf.reshape(next_layer.kernel, (-1, output_shape))
    kappa = tf.linalg.diag(kappa)
    # See Eq. 10. 
    W_fused = tf.matmul(kappa, W)
    W_fused = tf.reshape(W_fused, tf.shape(next_layer.kernel))    
    # See Eq. 9. 
    b_fused = next_layer.bias + tf.reduce_sum(tf.matmul(tf.linalg.diag(b_term), W), axis=0)
    layer = copy_layer(next_layer)
    if 'conv' in next_layer.name:
        # Set BN flag to 1 and BN_before_ReLU to 0.
        layer.set_weights([W_fused, np.array([1]), np.array([0])])
        # Create bias which will have 9 different values. 
        # Those values are generated by subtracting from b_fused the terms which come through padded input.
        # The obtained results is in Eq. 11.
        layer.set_bias(bias=b_fused, W=next_layer.kernel, b_term=b_term)
    else:
        layer.set_weights([W_fused, b_fused])
    fused_model.add(layer)
    if (i+1)!=len(model.layers)-1:
        fused_model.add(copy_layer(model.layers[i+2])) 
    return i+2