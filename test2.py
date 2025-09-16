import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # reduce TF verbosity
os.environ['CUDA_VISIBLE_DEVICES']='0'  #'-1' for CPU only
import argparse
import pickle as pkl
from Dataset import Dataset
from model import *
import time
from tensorflow.keras.layers import Lambda
from keras.layers import Activation
import tensorflow as tf
from tensorflow.keras.layers import ReLU, Lambda
import numpy as np
import pickle as pkl
import logging
from utils import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import os
from tensorflow.keras.callbacks import Callback





start_time = time.time()
# tf.keras.backend.set_floatx('float32') #to avoid numerical differences when comparing training of ReLU vs SNN
# tf.keras.backend.set_floatx('float32') #to avoid numerical differences when comparing training of ReLU vs SNN
override = None


strtobool = (lambda s: s=='True')
parser = argparse.ArgumentParser(description='TTFS')
parser.add_argument('--data_name', type=str, default='MNIST', help='(MNIST|CIFAR10|CIFAR100)')
parser.add_argument('--logging_dir', type=str, default='./logs/', help='Directory for logging')
parser.add_argument('--model_type', type=str, default='SNN', help='(SNN|ReLU)')
parser.add_argument('--model_name', type=str, default='BN', help='Should contain (FC2|VGG[BN]): e.g. VGG_BN_test1')
parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--epochs', type=int, default=100, help='Epochs. 0 -skip training')
parser.add_argument('--testing', type=strtobool, default=False, help='Execute testing.')
parser.add_argument('--load', type=str, default='False', help='Load before training. (True|False|custom_name.h5)')
parser.add_argument('--save', type=strtobool, default=False, help='Store after training.')
# Robustness parameters:fused_model
parser.add_argument('--findMax', type=strtobool, default=False, help='Store after training.')
parser.add_argument('--showSummmary', type=strtobool, default=True, help='Store after training.')
# Robustness parameters:fused_model

parser.add_argument('--noise', type=float, default=0.0, help='Noise std.dev.')
parser.add_argument('--time_bits', type=int, default=0, help='number of bits to represent time. 0 -disabled')
parser.add_argument('--weight_bits', type=int, default=0, help='number of bits to represent weights. 0 -disabled')
parser.add_argument('--latency_quantiles', type=float, default=0.0, help='Number of quantiles to take into account when calculating t_max. 0 -disabled')
parser.add_argument('--mode', type=str, default='', help='Ignore: A hack to address a bug in argsparse during debugging')
args = parser.parse_known_args(override)
if(len(args[1])>0):
    print("Warning: Ignored args", args[1])
args = args[0]
args.model_name = args.data_name + '-' + args.model_name
set_up_logging(args.logging_dir, args.model_name)
robustness_params={
    'noise':args.noise,
    'time_bits':args.time_bits,
    'weight_bits': args.weight_bits,
    'latency_quantiles':args.latency_quantiles
}

# Create data object
data = Dataset(
    args.data_name,
    args.logging_dir,
    flatten='FC' in args.model_name,
    ttfs_convert='SNN' in args.model_type,
    ttfs_noise=args.noise,
)
# Get optimizer for training.
optimizer = get_optimizer(args.lr)
model = None
logging.info("#### Creating the model ####")

if 'VGG' in args.model_name:
    # We consider one architecture, a 15-layer VGG-like network.
    if 'MNIST' in args.data_name: #MNIST / FMNIST
        layers2D=[64, 64, 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
        layers1D=[512, 512]
    else:  #other: CIFAR10, CIFAR100
        layers2D = [64, 64, 'pool', 128, 128, 'pool', 256, 256, 256, 'pool', 512, 512, 512, 'pool', 512, 512, 512, 'pool']
        layers1D=[512]
    kernel_size=(3,3)
    regularizer = None
    initializer = 'glorot_uniform' # keras default
    BN = 'BN' in args.model_name
    if not BN:  # resort to settings similar to the initial VGG paper
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
        regularizer = tf.keras.regularizers.L2(5e-4)
        initializer = 'he_uniform'
    if 'SNN' in args.model_type:
        model = create_vgg_model_SNN(layers2D, kernel_size, layers1D, data, optimizer, robustness_params=robustness_params,
                                     kernel_regularizer=regularizer, kernel_initializer=initializer)
    if 'ReLU' in args.model_type:
        model =  VGG16()
        # model = create_vgg_model_ReLU (layers2D, kernel_size, layers1D, data, BN=BN, optimizer=optimizer,
                                    #    kernel_regularizer=regularizer, kernel_initializer=initializer)
# if model is None:
#     print('Please specify a valid model. Exiting.')
#     exit(1)








from keras.datasets import cifar10



# fused_model = model
fused_model = fuse_bn_functional(model)








# test_input = (data.x_test[0:1] - data.p) / (data.q - data.p)
# original_output = model(test_input, training=False).numpy()
# fused_output    = fused_model(test_input, training=False).numpy()



# print("Original model output:", original_output)
# print("Fused model output:", fused_output)
# print("Max difference:", np.abs(original_output - fused_output).max())





























import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model

# Assuming you already have:
# model        -> original VGG16 model
# fused_model  -> fused Conv2D+BN model

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.0  # normalize

# Take a sample or use full test set
x_sample = x_test[:10]  # first 10 images

# Get predictions
orig_pred = model.predict(x_sample)
fused_pred = fused_model.predict(x_sample)

# Compare predictions
for i in range(len(x_sample)):
    print(f"Sample {i}:")
    print("Original prediction:", np.round(orig_pred[i], 4))
    print("Fused prediction:   ", np.round(fused_pred[i], 4))
    print("Max difference:     ", np.max(np.abs(orig_pred[i] - fused_pred[i])))
    print("Predicted class orig:", np.argmax(orig_pred[i]))
    print("Predicted class fused:", np.argmax(fused_pred[i]))
    print("-" * 40)

# Optional: compute overall max difference for all samples
max_diff = np.max(np.abs(orig_pred - fused_pred))
print("Overall max difference:", max_diff)

if args.showSummmary:
    logging.info(model.summary())
    logging.info(fused_model.summary())








# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # Ensure output directory exists
# os.makedirs("model_comparison_plots", exist_ok=True)

# # CIFAR-10 class names
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
#                 'dog', 'frog', 'horse', 'ship', 'truck']


# import matplotlib.pyplot as plt
# import numpy as np
# from tensorflow.keras.datasets import cifar10

# # Load CIFAR-10 dataset
# index = 2
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_test = x_test.astype('float32')

# # Extract a single image and its label
# test_image = x_test[index]
# true_label = y_test[index][0]

# # Prepare image for prediction
# test_image_batch = np.expand_dims(test_image, axis=0)
# predicted_prob = fused_model.predict(test_image_batch)
# predicted_class = np.argmax(predicted_prob)

# # Create subplots
# fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# # Plot the original image with true label
# axes[0].imshow(test_image.astype('uint8'))
# axes[0].set_title(f"True Label: {true_label}")
# axes[0].axis('off')

# # Plot the image with predicted class and true label
# axes[1].imshow(test_image.astype('uint8'))
# axes[1].set_title(f"Predicted: {predicted_class} | True: {true_label}")
# axes[1].axis('off')

# plt.tight_layout()
# plt.show()

# # Save image


# # # Prepare image for model input
# # test_input = (img - data.p) / (data.q - data.p)
# # test_input = np.expand_dims(test_input, axis=0)

# # # Get predictions
# # orig_pred = model(test_input, training=False).numpy().flatten()
# # fused_pred = fused_model(test_input, training=False).numpy().flatten()

# # # Save bar chart
# # x = np.arange(len(class_names))
# # width = 0.35
# # plt.figure(figsize=(8, 4))
# # plt.bar(x - width/2, orig_pred, width, label='Original')
# # plt.bar(x + width/2, fused_pred, width, label='Fused')
# # plt.xticks(x, class_names, rotation=45)
# # plt.ylabel('Probability')
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig(f"model_comparison_plots/prediction_comparison_{idx}.png")
# # plt.close()

# print(f"Saved image and prediction chart for test index  in 'model_comparison_plots/'")












