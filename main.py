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

class SaveWeightsEveryNEpochs(Callback):
    def __init__(self, save_path, n=10):
        super().__init__()
        self.save_path = save_path
        self.n = n
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.n == 0:
            filename = os.path.join(self.save_path, f'weights_epoch_{epoch + 1}.weights.h5')
            self.model.save_weights(filename)
            print(f'\nSaved weights at epoch {epoch + 1} to {filename}')




start_time = time.time()
# tf.keras.backend.set_floatx('float64') #to avoid numerical differences when comparing training of ReLU vs SNN
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
if 'FC2' in args.model_name:
    if 'SNN' in args.model_type:
        model = create_fc_model_SNN(layers=2, optimizer=optimizer, robustness_params=robustness_params)
    if 'ReLU' in args.model_type:
        model = create_fc_model_ReLU(layers=2, optimizer=optimizer)
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
        model = create_vgg_model_ReLU (layers2D, kernel_size, layers1D, data, BN=BN, optimizer=optimizer,
                                       kernel_regularizer=regularizer, kernel_initializer=initializer)
if model is None:
    print('Please specify a valid model. Exiting.')
    exit(1)
# model.summary()
model.last_dense = list(filter(lambda x : 'dense' in x.name, model.layers))[-1]

if args.load != 'False':
    logging.info("#### Loading weights ####")
    if 'ReLU' in args.model_type:
        # Load weights
        if args.load :  # automatic name
            print('path=', args.logging_dir +'/'+args.load)
            model.load_weights(args.logging_dir +'/'+args.load)
        # else:  # custom name
        #     model.load_weights(args.logging_dir + args.load, by_name=True)
    if 'SNN' in args.model_type:
        # Load X ranges
        #if os.path.exists(args.logging_dir + args.model_name + '_X_n.pkl'):
        X_n=pkl.load(open(args.logging_dir + args.model_name + '_X_n.pkl', 'rb'))
        # else:
        #     X_n=1000
        model.load_weights(args.logging_dir + args.model_name + '_preprocessed.h5', by_name=True)

if 'SNN' in args.model_type:
    logging.info("#### Setting SNN intervals ####")
    # Set parameters of SNN network: t_min_prev, t_min, t_max.
    t_min, t_max = 0, 1  # for the input layer
    for layer in model.layers:
        if 'conv' in layer.name or 'dense' in layer.name:
            t_min, t_max = layer.set_params(t_min, t_max)



logging.info("#### Training ####")


save_callback = SaveWeightsEveryNEpochs(save_path=args.logging_dir, n=1)

history = model.fit(
    data.x_train, data.y_train,
    batch_size=args.batch_size,
    epochs=args.epochs,
    verbose=1,
    validation_data=(data.x_test, data.y_test),
    callbacks=[save_callback]
)


if args.testing and args.epochs > 0:
    # Obtain accuracy of the fine-tuned SNN model.
    logging.info("#### Final test set accuracy testing ####")
    test_acc = model.evaluate(data.x_test, data.y_test, batch_size=args.batch_size)
    logging.info("Final testing accuracy is {}.".format(test_acc))

if args.save and 'ReLU' in args.model_type:
    # logging.info("#### Saving ReLU model ####")
 
    # model.save_weights(args.logging_dir + '/newTrain.weights.h5')
    fused_model = fuse_bn_functional(model)
    if args.testing:
      logging.info("#### Initial test set accuracy testing ####")
      test_model_acc = model.evaluate(data.x_test, data.y_test, batch_size=args.batch_size)
      test_fused_model_acc = fused_model.evaluate(data.x_test, data.y_test, batch_size=args.batch_size)
      logging.info("Initial testing model accuracy is {}.".format(test_model_acc))
      logging.info("Initial testing fused_model accuracy is {}.".format(test_fused_model_acc))
      
    logging.info('fuse (imaginary) BN layers')

    data.x_test, data.x_train = (data.x_test - data.p)/(data.q-data.p), (data.x_train - data.p)/(data.q-data.p)
    BN = 'BN' in args.model_name
  
    
    # test_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
    # original_output = model.predict(test_input,training=False)
    # fused_output = fused_model.predict(test_input,training=False)
    test_input = (data.x_test[0:1] - data.p) / (data.q - data.p)
    original_output = model(test_input, training=False).numpy()
    fused_output    = fused_model(test_input, training=False).numpy()



    print("Original model output:", original_output)
    print("Fused model output:", fused_output)
    print("Max difference:", np.abs(original_output - fused_output).max())
    from utils import fuse_bn_functional, verify_model_fusion
    is_correct = verify_model_fusion(model, fused_model)
    logging.info(f"Model fusion verified: {is_correct}")

    # import matplotlib.pyplot as plt
    # import numpy as np
    # import os

    # # Ensure output directory exists
    # os.makedirs("model_comparison_plots", exist_ok=True)

    # # CIFAR-10 class names
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
    #               'dog', 'frog', 'horse', 'ship', 'truck']

    # # Pick a test image index
    # idx = 0
    # img = data.x_test[idx]
    # label = np.argmax(data.y_test[idx])

    # # Undo normalization for display
    # img_display = img * (data.q - data.p) + data.p
    # img_display = np.clip(img_display, 0, 1)

    # # Save image
    # plt.figure(figsize=(2, 2))
    # plt.imshow(img_display)
    # plt.axis('off')
    # plt.title(f"True label: {class_names[label]}")
    # plt.savefig(f"model_comparison_plots/test_image_{idx}.png", bbox_inches='tight')
    # plt.close()

    # # Prepare image for model input
    # test_input = (img - data.p) / (data.q - data.p)
    # test_input = np.expand_dims(test_input, axis=0)

    # # Get predictions
    # orig_pred = model(test_input, training=False).numpy().flatten()
    # fused_pred = fused_model(test_input, training=False).numpy().flatten()

    # # Save bar chart
    # x = np.arange(len(class_names))
    # width = 0.35
    # plt.figure(figsize=(8, 4))
    # plt.bar(x - width/2, orig_pred, width, label='Original')
    # plt.bar(x + width/2, fused_pred, width, label='Fused')
    # plt.xticks(x, class_names, rotation=45)
    # plt.ylabel('Probability')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"model_comparison_plots/prediction_comparison_{idx}.png")
    # plt.close()

    # print(f"Saved image and prediction chart for test index {idx} in 'model_comparison_plots/'")





    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import os

    os.makedirs("layer_diffs", exist_ok=True)

    idx = 0
    test_input = np.expand_dims(data.x_test[idx], axis=0)

    # Submodels for all layer outputs
    orig_submodel = tf.keras.Model(inputs=model.input,
                                  outputs=[l.output for l in model.layers])
    fused_submodel = tf.keras.Model(inputs=fused_model.input,
                                    outputs=[l.output for l in fused_model.layers])

    orig_outputs = orig_submodel(test_input, training=False)
    fused_outputs = fused_submodel(test_input, training=False)

    diffs = []
    layer_names = []
    for o, f, l1, l2 in zip(orig_outputs, fused_outputs, model.layers, fused_model.layers):
        if o.shape == f.shape:  # compare only if shapes match
            diff = np.mean(np.abs(o.numpy() - f.numpy()))
            diffs.append(diff)
            layer_names.append(l1.name)
        else:
            print(f"Skipping {l1.name} vs {l2.name} due to shape mismatch: {o.shape} vs {f.shape}")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(range(len(diffs)), diffs, marker='o')
    plt.xticks(range(len(layer_names)), layer_names, rotation=90)
    plt.xlabel("Layer name")
    plt.ylabel("Mean absolute difference")
    plt.title("Layer-by-Layer Difference (Original vs Fused)")
    plt.tight_layout()
    plt.savefig("layer_diffs/layer_diff_plot.png")
    plt.close()

    print("Saved plot to layer_diffs/layer_diff_plot.png")







   
    
    logging.info(model.summary())
    logging.info(fused_model.summary())




    # Test with normalized input
    # sample = np.expand_dims(data.x_test[0], axis=0)  # Add batch dimension
    # output_original = model.predict(sample)
    # output_fused = model1.predict(sample)

    # def get_layer_outputs(model, input):
    #     outputs = {}
    #     x = input
    #     for name, layer in model.named_children():  # Works for Sequential
    #         x = layer(x)
    #         outputs[name] = x.detach()  # Store output
    #     return outputs

    # # Get outputs for both models
    # outputs_orig = get_layer_outputs(model, sample)
    # outputs_fused = get_layer_outputs(model1, sample)


    # Check if outputs are close
    # print("Outputs close?", np.allclose(output_original, output_fused, atol=1e-6))
    # 2. Save preprocessed ReLU model.
    model.save_weights(args.logging_dir + '/' + args.model_name + '_preprocessed.weights.h5')
    logging.info('saved preprocessed ReLU model')
    if args.findMax:
      # 3. Find maximum layer outputs.
      logging.info('calculating maximum layer output...')
      layer_num, X_n = 0, []
      layers_max = []
      for k, layer in enumerate(model.layers):
          if 'conv' in layer.name or 'dense' in layer.name:
              if k != len(model.layers) - 2:
                  # Apply ReLU first
                  relu_output = ReLU()(layer.output)
                  
                  # Wrap tf.reduce_max in a Lambda layer
                  max_output = Lambda(lambda x: tf.reduce_max(x))(relu_output)
                  
                  layers_max.append(max_output)

      extractor = tf.keras.Model(inputs=model.inputs, outputs=layers_max)
      output = extractor.predict(data.x_train, batch_size=64, verbose=1)
      X_n = list(map(lambda x: np.max(x), output))
      logging.info('X_n: %s', X_n)
      pkl.dump(X_n, open(args.logging_dir + '/' + args.model_name + '_X_n.pkl', 'wb'))
      logging.info('saved maximum layer output')

print('### Total elapsed time [s]:', time.time() - start_time)