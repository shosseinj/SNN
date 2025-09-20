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
from tensorflow.keras.layers import Lambda

from Dataset import Dataset
from model import *
from utils import set_up_logging, get_optimizer


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
            # TF native format (recommended)
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
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
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
# TRAINING FUNCTION
# ==============================
def main():
    start_time = time.time()
    args = parse_arguments()
    args.model_name = args.data_name + "-" + args.model_name

    # Setup logging
    set_up_logging(args.logging_dir, args.model_name)
    logging.info("### Starting Training Script ###")

    # TensorBoard setup
    log_dir = os.path.join("logs", args.model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

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
    optimizer = get_optimizer(args.lr)
    model = None

    # ==============================
    # MODEL CREATION
    # ==============================
    logging.info("#### Creating the model ####")
    if 'FC2' in args.model_name:
        if args.model_type == 'SNN':
            model = create_fc_model_SNN(layers=2, optimizer=optimizer, robustness_params=robustness_params)
        else:
            model = create_fc_model_ReLU(layers=2, optimizer=optimizer)

    elif 'VGG' in args.model_name:
        layers2D = [64, 64, 'pool', 128, 128, 'pool',
                    256, 256, 256, 'pool', 512, 512, 512, 'pool',
                    512, 512, 512, 'pool']
        layers1D = [512]
        kernel_size = (3, 3)
        BN = 'BN' in args.model_name
        regularizer = None
        initializer = 'glorot_uniform'

        if not BN:  # Original VGG settings
            optimizer = SGD(learning_rate=args.lr, momentum=0.9)
            regularizer = tf.keras.regularizers.L2(5e-4)
            initializer = 'he_uniform'

        if args.model_type == 'SNN':
            model = VGG_SNN(layers2D, kernel_size=(3,3), layers1D=layers1D, data=data, optimizer=optimizer, robustness_params=robustness_params)



# model = VGG_SNN(layers2D=[32, 64, 'pool'], layers1D=[128, 64], data=data)
# y_pred, min_ti = model(data.x_train[:4])
        else:
            model = VGG16()

    assert model is not None, "Model could not be created!"
    tf.config.run_functions_eagerly(True)
    # ==============================
    # TRAINING
    # ==============================




# ------------------------
# Training example


 



    if args.training:
        logging.info("#### Training ####")

        # Compile model
        num_dummy = 1
        dummy_loss = lambda y_true, y_pred: 0.0
        # model.compile(
        #     optimizer=Adam(),
        #     loss=[CategoricalCrossentropy(from_logits=True)] + [dummy_loss] * num_dummy,
        #     loss_weights=[1.0] + [0.0] * num_dummy,
        #     metrics={name: ['accuracy'] if i == 0 else [] for i, name in enumerate(model.output_names)}
        # )

        # Dummy outputs
        dummy_train = np.zeros((data.x_train.shape[0], num_dummy))
        dummy_test = np.zeros((data.x_test.shape[0], num_dummy))

        # Callbacks
        save_cb = SaveWeightsEveryNEpochs("weights/", n=5)
        checkpoint_cb = ModelCheckpoint(
            "weights/best_model",      # no .h5 extension
            save_best_only=True,
            monitor="val_loss",
            save_weights_only=True     # ⚠ important
        )
      
      

        model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        history = model.fit(
    data.x_train,
    data.y_train,
    batch_size=args.batch_size,
    epochs=args.epochs,
    validation_data=(data.x_test, data.y_test),
    callbacks=[tensorboard_cb, save_cb, checkpoint_cb],   # <-- add callbacks here
    verbose=1
)


        # history = model.fit(
        #     data.x_train,
        #     [data.y_train] + [dummy_train] * num_dummy,
        #     batch_size=args.batch_size,
        #     epochs=args.epochs,
        #     validation_data=(data.x_test, [data.y_test] + [dummy_test] * num_dummy),
        #     callbacks=[tensorboard_cb, save_cb, checkpoint_cb],
        #     verbose=1
        # )

        # Evaluate
        test_loss, test_acc, *_ = model.evaluate(data.x_test, [data.y_test, dummy_test], verbose=1)
        logging.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

        if args.save:
            model.save_weights("weights/final_model")  # TF format
            logging.info("Model saved → weights/final_model.h5")

    # ==============================
    # OPTIONAL: Find max activations
    # ==============================
    if args.findMax:
        logging.info("#### Finding Maximum Layer Outputs ####")
        layers_max = []
        for k, layer in enumerate(model.layers):
            if ('conv' in layer.name or 'dense' in layer.name) and k != len(model.layers) - 2:
                relu_max_layer = Lambda(lambda x: tf.reduce_max(tf.nn.relu(x)))(layer.output)
                layers_max.append(relu_max_layer)

        extractor = Model(inputs=model.inputs, outputs=layers_max)
        output = extractor.predict(data.x_train, batch_size=64, verbose=1)
        X_n = [np.max(x) for x in output]

        with open("weights/RELU_X_n.pkl", "wb") as f:
            pkl.dump(X_n, f)

        logging.info("Saved maximum layer outputs → weights/RELU_X_n.pkl")

    logging.info(f"### Total Elapsed Time: {time.time() - start_time:.2f} sec ###")


if __name__ == "__main__":
    main()
