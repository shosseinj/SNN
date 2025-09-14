import numpy as np
import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(
        self,
        data_name,
        logging_dir,
        flatten,
        ttfs_convert,
        ttfs_noise=0,
    ):
        self.name = data_name
        self.flatten=flatten
        self.noise=ttfs_noise
        self.logging_dir=logging_dir
        # Load original data.
        self.get_features_vectors()
        # In case of SNN, convert input data with TTFS coding.
        self.ttfss_convert=ttfs_convert
        if ttfs_convert: self.convert_ttfs()
        
    def get_features_vectors(self):
        """
        Load image datasets and transform into features. 
        """

        if 'CIFAR' in self.name:
            # CIFAR10 or CIFAR100 dataset.
            self.input_shape=(32, 32, 3)
            self.q, self.p = 3.0, -3.0
            if self.name=='CIFAR10':
                # CIFAR10 dataset.
                self.num_of_classes = 10
                (self.x_train,self.y_train), (self.x_test,self.y_test)=tf.keras.datasets.cifar10.load_data()
                # Mean and std to scale input.
                self.mean_test, self.std_test=120.707, 64.15
            else:
                # CIFAR100
                self.num_of_classes = 100
                (self.x_train,self.y_train), (self.x_test,self.y_test)=tf.keras.datasets.cifar100.load_data()
                # Mean and std to scale input.
                self.mean_test, self.std_test=121.936, 68.389
            # Scale to [-3, 3] range.
            self.x_test, self.x_train=(self.x_test-self.mean_test)/(self.std_test+1e-7), (self.x_train-self.mean_test)/(self.std_test+1e-7)
        # Processing which is the same for all datasets.
        self.x_train, self.x_test = self.x_train.astype('float32'), self.x_test.astype('float32')
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_of_classes)
        self.y_train = tf.keras.utils.to_categorical(self.y_train, self.num_of_classes)
        print ('Train data:', np.shape(self.x_train), np.shape(self.y_train))
        print ('Test data:', np.shape(self.x_test), np.shape(self.y_test))

    # def convert_ttfs(self):
    #     """
    #     Convert input values into time-to-first-spike spiking times.
    #     """
    #     self.x_test, self.x_train = (self.x_test - self.p)/(self.q-self.p), (self.x_train - self.p)/(self.q-self.p)
    #     self.x_train, self.x_test=1 - np.array(self.x_train), 1 - np.array(self.x_test)
    #     self.x_test=np.maximum(0, self.x_test + tf.random.normal((self.x_test).shape, stddev=self.noise, dtype=tf.dtypes.float32))
    def convert_ttfs(self):
        """
        Convert input values into time-to-first-spike spiking times.
        """
        # Scale to [0,1]
        self.x_train = (self.x_train - self.p) / (self.q - self.p)
        self.x_test  = (self.x_test  - self.p) / (self.q - self.p)

        # Flip (1 - x), keep float32
        self.x_train = 1.0 - self.x_train
        self.x_test  = 1.0 - self.x_test

        # Add noise if needed
        if self.noise > 0:
            noise = tf.random.normal(self.x_test.shape, stddev=self.noise, dtype=tf.float32)
            self.x_test += noise
            np.clip(self.x_test, 0, None, out=self.x_test)  # in-place clipping

