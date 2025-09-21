# import os
# import time
# import argparse
# import logging
# import datetime
# import numpy as np
# import pickle as pkl
# import tensorflow as tf

# from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint
# from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Lambda

# from Dataset import Dataset
# from model import *
# from utils import set_up_logging, get_optimizer


# # ==============================
# # GPU CONFIGURATION
# # ==============================
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce TF verbosity
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set GPU (use '-1' for CPU only)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# tf.keras.backend.set_floatx('float32')


# # ==============================
# # CUSTOM CALLBACK
# # ==============================
# class SaveWeightsEveryNEpochs(Callback):
#     def __init__(self, save_path, n=10):
#         super().__init__()
#         self.save_path = save_path
#         self.n = n
#         os.makedirs(save_path, exist_ok=True)

#     def on_epoch_end(self, epoch, logs=None):
#         if (epoch + 1) % self.n == 0:
#             filename = os.path.join(self.save_path, f'weights_epoch_{epoch + 1}')
#             # TF native format (recommended)
#             self.model.save_weights(filename)
#             print(f"\n[INFO] Saved weights at epoch {epoch + 1} → {filename}")



# # ==============================
# # ARGUMENT PARSER
# # ==============================
# def parse_arguments():
#     strtobool = lambda s: s.lower() in ['true', '1', 'yes']
#     parser = argparse.ArgumentParser(description="SNN/ReLU Training Script")

#     parser.add_argument('--data_name', type=str, default='MNIST', help='Dataset: MNIST | CIFAR10 | CIFAR100')
#     parser.add_argument('--logging_dir', type=str, default='./logs/', help='Directory for logging')
#     parser.add_argument('--model_type', type=str, default='SNN', help='Model type: SNN | ReLU')
#     parser.add_argument('--model_name', type=str, default='BN', help='Model name (contains FC2 or VGG)')
#     parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
#     parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
#     parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
#     parser.add_argument('--training', type=strtobool, default=True, help='Enable training mode')
#     parser.add_argument('--testing', type=strtobool, default=False, help='Enable testing mode')
#     parser.add_argument('--save', type=strtobool, default=True, help='Save model after training')
#     parser.add_argument('--load', type=str, default='False', help='Load pre-trained weights')
#     parser.add_argument('--findMax', type=strtobool, default=False, help='Find maximum activations per layer')

#     # Robustness parameters
#     parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
#     parser.add_argument('--time_bits', type=int, default=0, help='Quantization bits for time')
#     parser.add_argument('--weight_bits', type=int, default=0, help='Quantization bits for weights')
#     parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Quantile for latency')

#     args, unknown = parser.parse_known_args()
#     if unknown:
#         print(f"[WARNING] Ignored args: {unknown}")
#     return args


# # ==============================
# # TRAINING FUNCTION
# # ==============================
# def main():
#     start_time = time.time()
#     args = parse_arguments()
#     args.model_name = args.data_name + "-" + args.model_name

#     # Setup logging
#     set_up_logging(args.logging_dir, args.model_name)
#     logging.info("### Starting Training Script ###")

#     # TensorBoard setup
#     log_dir = os.path.join("logs", args.model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#     tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

#     # Robustness parameters
#     robustness_params = {
#         'noise': args.noise,
#         'time_bits': args.time_bits,
#         'weight_bits': args.weight_bits,
#         'latency_quantiles': args.latency_quantiles
#     }

#     # Dataset
#     data = Dataset(
#         args.data_name,
#         args.logging_dir,
#         flatten='FC' in args.model_name,
#         ttfs_convert='SNN' in args.model_type,
#         ttfs_noise=args.noise,
#     )

#     # Optimizer
#     optimizer = get_optimizer(args.lr)
#     model = None

#     # ==============================
#     # MODEL CREATION
#     # ==============================
#     logging.info("#### Creating the model ####")
#     if 'FC2' in args.model_name:
#         if args.model_type == 'SNN':
#             model = create_fc_model_SNN(layers=2, optimizer=optimizer, robustness_params=robustness_params)
#         else:
#             model = create_fc_model_ReLU(layers=2, optimizer=optimizer)

#     elif 'VGG' in args.model_name:
#         layers2D = [64, 64, 'pool', 128, 128, 'pool',
#                     256, 256, 256, 'pool', 512, 512, 512, 'pool',
#                     512, 512, 512, 'pool']
#         layers1D = [512]
#         kernel_size = (3, 3)
#         BN = 'BN' in args.model_name
#         regularizer = None
#         initializer = 'glorot_uniform'

#         if not BN:  # Original VGG settings
#             optimizer = SGD(learning_rate=args.lr, momentum=0.9)
#             regularizer = tf.keras.regularizers.L2(5e-4)
#             initializer = 'he_uniform'

#         if args.model_type == 'SNN':
#             model = VGG_SNN(layers2D, kernel_size=(3,3), layers1D=layers1D, data=data, optimizer=optimizer, robustness_params=robustness_params)



# # model = VGG_SNN(layers2D=[32, 64, 'pool'], layers1D=[128, 64], data=data)
# # y_pred, min_ti = model(data.x_train[:4])
#         else:
#             model = VGG16()

#     assert model is not None, "Model could not be created!"
#     tf.config.run_functions_eagerly(True)
#     # ==============================
#     # TRAINING
#     # ==============================




# # ------------------------
# # Training example


 



#     if args.training:
#         logging.info("#### Training ####")

#         # Compile model
#         num_dummy = 1
#         dummy_loss = lambda y_true, y_pred: 0.0
#         # model.compile(
#         #     optimizer=Adam(),
#         #     loss=[CategoricalCrossentropy(from_logits=True)] + [dummy_loss] * num_dummy,
#         #     loss_weights=[1.0] + [0.0] * num_dummy,
#         #     metrics={name: ['accuracy'] if i == 0 else [] for i, name in enumerate(model.output_names)}
#         # )

#         # Dummy outputs
#         dummy_train = np.zeros((data.x_train.shape[0], num_dummy))
#         dummy_test = np.zeros((data.x_test.shape[0], num_dummy))

#         # Callbacks
#         save_cb = SaveWeightsEveryNEpochs("weights/", n=5)
#         checkpoint_cb = ModelCheckpoint(
#             "weights/best_model",      # no .h5 extension
#             save_best_only=True,
#             monitor="val_loss",
#             save_weights_only=True     # ⚠ important
#         )
      
      

#         model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#         history = model.fit(
#     data.x_train,
#     data.y_train,
#     batch_size=args.batch_size,
#     epochs=args.epochs,
#     validation_data=(data.x_test, data.y_test),
#     callbacks=[tensorboard_cb, save_cb, checkpoint_cb],   # <-- add callbacks here
#     verbose=1
# )


#         # history = model.fit(
#         #     data.x_train,
#         #     [data.y_train] + [dummy_train] * num_dummy,
#         #     batch_size=args.batch_size,
#         #     epochs=args.epochs,
#         #     validation_data=(data.x_test, [data.y_test] + [dummy_test] * num_dummy),
#         #     callbacks=[tensorboard_cb, save_cb, checkpoint_cb],
#         #     verbose=1
#         # )

#         # Evaluate
#         test_loss, test_acc, *_ = model.evaluate(data.x_test, [data.y_test, dummy_test], verbose=1)
#         logging.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

#         if args.save:
#             model.save_weights("weights/final_model")  # TF format
#             logging.info("Model saved → weights/final_model.h5")

#     # ==============================
#     # OPTIONAL: Find max activations
#     # ==============================
#     if args.findMax:
#         logging.info("#### Finding Maximum Layer Outputs ####")
#         layers_max = []
#         for k, layer in enumerate(model.layers):
#             if ('conv' in layer.name or 'dense' in layer.name) and k != len(model.layers) - 2:
#                 relu_max_layer = Lambda(lambda x: tf.reduce_max(tf.nn.relu(x)))(layer.output)
#                 layers_max.append(relu_max_layer)

#         extractor = Model(inputs=model.inputs, outputs=layers_max)
#         output = extractor.predict(data.x_train, batch_size=64, verbose=1)
#         X_n = [np.max(x) for x in output]

#         with open("weights/RELU_X_n.pkl", "wb") as f:
#             pkl.dump(X_n, f)

#         logging.info("Saved maximum layer outputs → weights/RELU_X_n.pkl")

#     logging.info(f"### Total Elapsed Time: {time.time() - start_time:.2f} sec ###")


# if __name__ == "__main__":
#     main()


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
from tensorflow.keras.models import Model
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
# UTILITY FUNCTIONS (Placeholders)
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
            filename = os.path.join(self.save_path, f'weights_epoch_{epoch + 1}')
            self.model.save_weights(filename)
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



    """Create a simple fully connected ReLU model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

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




    model.load_weights('./cifar10vgg.h5')

    return model

# ==============================
# SNN LAYERS
# ==============================
def call_spiking(tj, W, D_i, t_min_prev, t_min, t_max, robustness_params):
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
            initializer='glorot_uniform', trainable=True, name='kernel',     regularizer=tf.keras.regularizers.l2(5e-4),  # ✅ add this

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
    def __init__(self, filters, kernel_size=(3,3), X_n=1, padding='same',dropout_rate=0.2, robustness_params={}, name=None):
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.B_n = (1 + 0.5) * X_n
        self.robustness_params = robustness_params
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        in_channels = input_shape[-1]  # use real input channels
        self.kernel = self.add_weight(
            shape=(self.kernel_size[0], self.kernel_size[1], in_channels, self.filters),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel',
            regularizer=tf.keras.regularizers.l2(5e-4)
        )
        self.D_i = self.add_weight(shape=(9, self.filters), initializer='zeros', trainable=True, name='D_i')
        self.t_min_prev = tf.Variable(0.0, trainable=False)
        self.t_min = tf.Variable(0.0, trainable=False)
        self.t_max = tf.Variable(1.0, trainable=False)

    def call(self, tj, training=None):
        padding_size = (self.kernel_size[0] // 2) if self.padding=='same' else 0
        image_size = tf.shape(tj)[1]
        # tj = tf.pad(tj, [[0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0]], 
        #            constant_values=self.t_min.numpy())
        tj = tf.pad(
             tj,
            [[0,0],[padding_size,padding_size],[padding_size,padding_size],[0,0]], 
            constant_values=tf.cast(self.t_min, tj.dtype)  # ✅ works in graph mode
        )
        tj_patches = tf.image.extract_patches(
            tj, sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1,1,1,1], rates=[1,1,1,1], padding='VALID'
        )
        W = tf.reshape(self.kernel, (-1, self.filters))
        tj_flat = tf.reshape(tj_patches, (-1, tf.shape(W)[0]))
        ti = call_spiking(tj_flat, W, self.D_i[0], self.t_min_prev, self.t_min, self.t_max, self.robustness_params)
        ti = tf.reshape(ti, (-1, image_size, image_size, self.filters))
        # ti = self.dropout(ti, training=training)    
        return ti

class MaxMinPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
    
    def call(self, x):
        x = tf.nn.max_pool2d(x, ksize=self.pool_size, strides=self.pool_size, padding='SAME')
        return x

class VGG_SNN(tf.keras.Model):
    def __init__(self, layers2D, kernel_size, layers1D, data, optimizer, robustness_params={}):
        super().__init__()
        self.conv_layers = []
        # Use simplified architecture for testing
        for i, f in enumerate(layers2D):  # Simplified for testing
            if f != 'pool':
                self.conv_layers.append(SpikingConv2D(f, kernel_size=kernel_size, X_n=1000,
                                                     robustness_params=robustness_params, name=f'conv_{i+1}'))
            else:
                self.conv_layers.append('pool')
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layers = [SpikingDense(128, X_n=1000, robustness_params=robustness_params, name='dense_1')]
        self.output_layer = SpikingDense(data.num_of_classes, outputLayer=True, 
                                       robustness_params=robustness_params, name='dense_out')
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
        ttfs_convert='SNN' in args.model_type,
        ttfs_noise=args.noise,
    )

    # Optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.lr,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)
    optimizer = Adam(learning_rate=lr_schedule)
    model = None

    # Model creation
    logging.info("#### Creating the model ####")

    if 'VGG' in args.model_name:
        layers2D = [64, 64, 'pool', 128, 128, 'pool',
                    256, 256, 256, 'pool', 512, 512, 512, 'pool',
                    512, 512, 512, 'pool']
        layers1D = [512]
        kernel_size = (3, 3)
        # model.build((None,) + data.input_shape)
# model.summary()
        model = VGG_SNN(layers2D, kernel_size, layers1D, data, optimizer, robustness_params)
        model_ann = VGG16(input_shape=data.input_shape, classes=data.num_of_classes)
        dummy_input = tf.random.normal((1,) + data.input_shape)  # e.g., (1, 32, 32, 3)
        model(dummy_input)  # this calls build() for all layers

        # Now you can assign weights
        for ann_layer, snn_layer in zip(model_ann.layers, model.conv_layers):
            if isinstance(ann_layer, Conv2D) and isinstance(snn_layer, SpikingConv2D):
                W = ann_layer.get_weights()[0]
                W = W / np.max(np.abs(W))
                if W.shape == snn_layer.kernel.shape:
                    snn_layer.kernel.assign(W)
                else:
                    print(f"[WARNING] Skipping {ann_layer.name} due to shape mismatch")


    
    # Enable eager execution for debugging
    tf.config.run_functions_eagerly(True)

    # Training
    if args.training:
        logging.info("#### Training ####")

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Callbacks
        save_cb = SaveWeightsEveryNEpochs("weights/", n=5)
        checkpoint_cb = ModelCheckpoint(
            "weights/best_model",
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True
        )
        tensorboard_cb = TensorBoard(
            log_dir=os.path.join("logs", args.model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        )

        # Test with small dataset first
        logging.info("Testing with small batch...")
        try:
            history = model.fit(
                data.x_train[:100],
                data.y_train[:100],
                batch_size=32,
                epochs=1,
                validation_data=(data.x_test[:50], data.y_test[:50]),
                verbose=1
            )
            logging.info("Small batch training successful!")
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return

        # Full training
        logging.info("Starting full training...")
        history = model.fit(
            data.x_train,
            data.y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(data.x_test, data.y_test),
            callbacks=[tensorboard_cb, save_cb, checkpoint_cb],
            verbose=1
        )

        # Evaluate
        test_loss, test_acc = model.evaluate(data.x_test, data.y_test, verbose=1)
        logging.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

        if args.save:
            model.save_weights("weights/final_model")
            logging.info("Model saved → weights/final_model")

    logging.info(f"### Total Elapsed Time: {time.time() - start_time:.2f} sec ###")

if __name__ == "__main__":
    main()