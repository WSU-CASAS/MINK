from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import tensorview as tv

import joblib
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv1D, Conv2D, Flatten, GRU
from tensorflow.keras.layers import Reshape, Conv2DTranspose, UpSampling1D
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow.keras.models import Model
import math
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse

import sys


def PCA_Analysis(dataX, dataX_hat):
    # Analysis Data Size
    Sample_No = min(len(dataX) - 100, len(dataX_hat) - 100)

    # Data Preprocessing
    for i in range(Sample_No):
        if i == 0:
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]), 1), [1, len(dataX[0][:, 0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]), 1), [1, len(dataX[0][:, 0])])
        else:
            arrayX = np.concatenate(
                (arrayX, np.reshape(np.mean(np.asarray(dataX[i]), 1), [1, len(dataX[0][:, 0])])))
            arrayX_hat = np.concatenate((arrayX_hat,
                                         np.reshape(np.mean(np.asarray(dataX_hat[i]), 1),
                                                    [1, len(dataX[0][:, 0])])))

    # Parameters
    No = len(arrayX[:, 0])
    colors = ["red" for i in range(No)] + ["blue" for i in range(No)]

    # PCA Analysis
    pca = PCA(n_components=2)
    pca.fit(arrayX)
    pca_results = pca.transform(arrayX)
    pca_hat_results = pca.transform(arrayX_hat)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(pca_results[:, 0], pca_results[:, 1], c=colors[:No], alpha=0.2, label="Original")
    plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], c=colors[No:], alpha=0.2,
                label="Synthetic")

    ax.legend()

    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()


def tSNE_Analysis(dataX, dataX_hat):
    # Analysis Data Size
    Sample_No = min(len(dataX) - 100, len(dataX_hat) - 100)

    # Preprocess
    for i in range(Sample_No):
        if i == 0:
            arrayX = np.reshape(np.mean(np.asarray(dataX[0]), 1), [1, len(dataX[0][:, 0])])
            arrayX_hat = np.reshape(np.mean(np.asarray(dataX_hat[0]), 1), [1, len(dataX[0][:, 0])])
        else:
            arrayX = np.concatenate(
                (arrayX, np.reshape(np.mean(np.asarray(dataX[i]), 1), [1, len(dataX[0][:, 0])])))
            arrayX_hat = np.concatenate((arrayX_hat,
                                         np.reshape(np.mean(np.asarray(dataX_hat[i]), 1),
                                                    [1, len(dataX[0][:, 0])])))

    # Do t-SNE Analysis together
    final_arrayX = np.concatenate((arrayX, arrayX_hat), axis=0)

    # Parameters
    No = len(arrayX[:, 0])
    colors = ["red" for i in range(No)] + ["blue" for i in range(No)]

    # TSNE anlaysis
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(final_arrayX)

    # Plotting
    f, ax = plt.subplots(1)

    plt.scatter(tsne_results[:No, 0], tsne_results[:No, 1], c=colors[:No], alpha=0.2,
                label="Original")
    plt.scatter(tsne_results[No:, 0], tsne_results[No:, 1], c=colors[No:], alpha=0.2,
                label="Synthetic")

    ax.legend()

    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()


def end_cond(X_train):
    print('end_cond()')
    print('X_train.shape', X_train.shape)
    vals = X_train[:, X_train.shape[1] - 1, 4] / X_train[:, 0, 4] - 1
    print('vals.shape', vals.shape)

    comb1 = np.where(vals < -.1, 0, 0)
    comb2 = np.where((vals >= -.1) & (vals <= -.05), 1, 0)
    comb3 = np.where((vals >= -.05) & (vals <= -.0), 2, 0)
    comb4 = np.where((vals > 0) & (vals <= 0.05), 3, 0)
    comb5 = np.where((vals > 0.05) & (vals <= 0.1), 4, 0)
    comb6 = np.where(vals > 0.1, 5, 0)
    cond_all = comb1 + comb2 + comb3 + comb4 + comb5 + comb6
    print('cond_all.shape', cond_all.shape)

    print(np.unique(cond_all, return_counts=True))
    arr = np.repeat(cond_all, X_train.shape[1], axis=0).reshape(len(cond_all), X_train.shape[1])
    print('arr.shape', arr.shape)
    X_train = np.dstack((X_train, arr))
    print('X_train.shape', X_train.shape)
    return X_train


def MinMax(train_data):
    scaler = MinMaxScaler()
    num_instances, num_time_steps, num_features = train_data.shape
    train_data = np.reshape(train_data, (-1, num_features))
    train_data = scaler.fit_transform(train_data)
    train_data = np.reshape(train_data, (num_instances, num_time_steps, num_features))
    return train_data, scaler


def google_data_loading(seq_length):
    print('google_data_loading()')
    # Load Google Data
    x = np.loadtxt('https://github.com/firmai/tsgan/raw/master/alg/timegan/data/GOOGLE_BIG.csv',
                   delimiter=",", skiprows=1)
    # x = labels("https://github.com/firmai/tsgan/raw/master/alg/timegan/data/GOOGLE_BIG.csv")
    x = np.array(x)
    print(x.shape)

    # Build dataset
    dataX = []

    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
    print('len dataX = ', len(dataX))
    print('len dataX[0] = ', len(dataX[0]))

    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))

    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])

    X_train = np.stack(dataX)
    print(X_train.shape)

    X_train = end_cond(X_train)  # Adding the conditions

    print('before minmax')
    print(X_train.shape)
    x_x, scaler = MinMax(X_train[:, :, :-1])
    print('x_x shape', x_x.shape)

    # x_x = np.c_[x_x, X_train[:,:,-1]]
    x_x = np.dstack((x_x, X_train[:, :, -1]))
    print('x_x shape', x_x.shape)

    return x_x, X_train, scaler


class MinkMSGAN:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.batch_size = 64  # Source had 64, our other GAN has 128
        self.train_steps = 100  # Set to 10000 for production
        self.columns = list(['yaw', 'pitch', 'roll', 'rotation_rate_x', 'rotation_rate_y',
                             'rotation_rate_z', 'user_acceleration_x', 'user_acceleration_y',
                             'user_acceleration_z', 'latitude', 'longitude', 'altitude', 'course',
                             'speed', 'horizontal_accuracy', 'vertical_accuracy'])
        self.n_seq = len(self.columns)
        self.shape = (self.seq_len, self.n_seq)
        self.gen0 = None
        self.gen1 = None
        self.scaler = None
        self.synthetic_data = None  # will hold synthetic data generator model
        self.use_random_z = False  # use random(True) or previous sequence(False) as input to model
        self._check_gpu()
        return

    def _make_random_data(self):
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))
        return
    
    def _check_gpu(self):
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        if gpu_devices:
            print('Using GPU')
        else:
            print('Using CPU')
        return
    
    def _make_rnn(self, n_layers, hidden_units, output_units, name):
        return Sequential([GRU(units=hidden_units,
                               return_sequences=True,
                               name=f'GRU_{i + 1}') for i in range(n_layers)] +
                               [Dense(units=output_units,
                                      activation='sigmoid',
                                      name='OUT')], name=name)

    def generator(self, inputs,
                  activation='sigmoid',
                  labels=None,
                  codes=None):
        """Build a Generator Model
        Output activation is sigmoid instead of tanh in as Sigmoid converges easily.
        [1] Radford, Alec, Luke Metz, and Soumith Chintala.
        "Unsupervised representation learning with deep convolutional
        generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015)..
        Arguments:
            inputs (Layer): Input layer of the generator (the z-vector)
            activation (string): Name of output activation layer
            codes (list): 2-dim disentangled codes
        Returns:
            Model: Generator Model
        """

        if codes is not None:
            # generator 0 of MTSS
            inputs = [inputs, codes]
            x = concatenate(inputs, axis=1)
            # noise inputs + conditional codes
        else:
            # default input is just a noise dimension (z-code)
            x = inputs

        x = Dense(self.shape[0] * self.shape[1])(x)
        x = Reshape((self.shape[0], self.shape[1]))(x)
        x = GRU(int((self.shape[0] * self.shape[1]) / 2),
                return_sequences=False,
                return_state=False,
                unroll=True)(x)
        x = Reshape((int(self.shape[0] / 2), self.shape[1]))(x)
        x = Conv1D(128, 4, 1, "same")(x)
        x = BatchNormalization(momentum=0.8)(x)  # adjusting and scaling the activations
        x = ReLU()(x)
        x = UpSampling1D()(x)
        x = Conv1D(self.n_seq, 4, 1, "same")(x)
        x = BatchNormalization(momentum=0.8)(x)

        if activation is not None:
            x = Activation(activation)(x)

        # generator output is the synthesized data x
        return Model(inputs, x, name='gen1')

    def discriminator(self, inputs,
                      activation='sigmoid',
                      num_labels=None,
                      num_codes=None):
        """Build a Discriminator Model
        The network does not converge with batch normalisation so it is not used here
        Arguments:
            inputs (Layer): Input layer of the discriminator (the sample)
            activation (string): Name of output activation layer
            num_codes (int): num_codes-dim Q network as output
                        if MTSS-GAN or 2 Q networks if InfoGAN

        Returns:
            Model: Discriminator Model
        """
        ints = int(self.shape[0] / 2)
        print('ints', ints)
        x = inputs
        x = GRU(self.shape[1] * self.shape[0],
                return_sequences=False,
                return_state=False,
                unroll=True,
                activation="relu")(x)
        x = Reshape((self.shape[0], self.shape[1]))(x)
        x = Conv1D(16, 3, 2, "same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1D(32, 3, 2, "same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1D(64, 3, 2, "same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1D(128, 3, 1, "same")(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        # default output is probability that the time series array is real
        outputs = Dense(1)(x)

        if num_codes is not None:
            # MTSS-GAN Q0 output
            # z0_recon is reconstruction of z0 normal distribution
            # eventually two loss functions from this output.
            z0_recon = Dense(num_codes)(x)
            z0_recon = Activation('tanh', name='z0')(z0_recon)
            outputs = [outputs, z0_recon]

        return Model(inputs, outputs, name='discriminator')

    def build_encoder(self, inputs, num_labels=6, feature0_dim=None):
        """ Build the Classifier (Encoder) Model sub networks
        Two sub networks:
        1) Encoder0: time series array to feature0 (intermediate latent feature)
        2) Encoder1: feature0 to labels
        # Arguments
            inputs (Layers): x - time series array, feature1 -
                feature1 layer output
            num_labels (int): number of class labels
            feature0_dim (int): feature0 dimensionality
        # Returns
            enc0, enc1 (Models): Description below
        """

        if feature0_dim is None:
            feature0_dim = self.shape[0] * self.shape[1]

        x, feature0 = inputs

        y = GRU(self.shape[0] * self.shape[1],
                return_sequences=False,
                return_state=False,
                unroll=True)(x)
        y = Flatten()(y)
        feature0_output = Dense(feature0_dim, activation='relu')(y)
        # Encoder0 or enc0: data to feature0
        enc0 = Model(inputs=x, outputs=feature0_output, name="encoder0")

        # Encoder1 or enc1
        y = Dense(num_labels)(feature0)
        labels = Activation('softmax')(y)
        # Encoder1 or enc1: feature0 to class labels
        enc1 = Model(inputs=feature0, outputs=labels, name="encoder1")

        # return both enc0 and enc1
        return enc0, enc1

    def build_generator(self, latent_codes, feature0_dim=None):
        """Build Generator Model sub networks
        Two sub networks: 1) Class and noise to feature0
            (intermediate feature)
            2) feature0 to time series array
        # Arguments
            latent_codes (Layers): dicrete code (labels),
                noise and feature0 features
            feature0_dim (int): feature0 dimensionality
        # Returns
            gen0, gen1 (Models): Description below
        """

        if feature0_dim is None:
            feature0_dim = self.shape[0] * self.shape[1]

        # Latent codes and network parameters
        labels, z0, z1, feature0 = latent_codes

        # gen0 inputs
        inputs = [labels, z0]  # 6 + 50 = 62-dim
        x = concatenate(inputs, axis=1)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        fake_feature0 = Dense(feature0_dim, activation='relu')(x)

        # gen0: classes and noise (labels + z0) to feature0
        gen0 = Model(inputs, fake_feature0, name='gen0')

        # gen1: feature0 + z0 to feature1 (time series array)
        # example: Model([feature0, z0], (steps, feats),  name='gen1')
        gen1 = self.generator(feature0, codes=z1)

        return gen0, gen1

    def build_discriminator(self, inputs, z_dim=50):
        """Build Discriminator 1 Model
        Classifies feature0 (features) as real/fake time series array and recovers
        the input noise or latent code (by minimizing entropy loss)
        # Arguments
            inputs (Layer): feature0
            z_dim (int): noise dimensionality
        # Returns
            dis0 (Model): feature0 as real/fake recovered latent code
        """

        # input is 256-dim feature1
        x = Dense(self.shape[0] * self.shape[1], activation='relu')(inputs)

        x = Dense(self.shape[0] * self.shape[1], activation='relu')(x)

        # first output is probability that feature0 is real
        f0_source = Dense(1)(x)
        f0_source = Activation('sigmoid',
                               name='feature1_source')(f0_source)

        # z0 reonstruction (Q0 network)
        z0_recon = Dense(z_dim)(x)
        z0_recon = Activation('tanh', name='z0')(z0_recon)

        discriminator_outputs = [f0_source, z0_recon]
        dis0 = Model(inputs, discriminator_outputs, name='dis0')
        return dis0

    def train(self, models, data, params):
        """Train the discriminator and adversarial Networks
        Alternately train discriminator and adversarial networks by batch.
        Discriminator is trained first with real and fake time series array,
        corresponding one-hot labels and latent codes.
        Adversarial is trained next with fake time series array pretending
        to be real, corresponding one-hot labels and latent codes.
        Generate sample time series data per save_interval.
        # Arguments
            models (Models): Encoder, Generator, Discriminator,
                Adversarial models
            data (tuple): x_train, y_train data
            params (tuple): Network parameters
        """
        # the MTSS-GAN and Encoder models

        enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1 = models
        # network parameters
        batch_size, train_steps, num_labels, z_dim, model_name = params
        # train dataset
        (x_train, y_train), (_, _) = data  # I can do this.
        # the generated time series array is saved every 500 steps
        save_interval = 500

        # label and noise codes for generator testing
        z0 = np.random.normal(scale=0.5, size=[self.shape[0], z_dim])
        z1 = np.random.normal(scale=0.5, size=[self.shape[0], z_dim])
        noise_class = np.eye(num_labels)[np.arange(0, self.shape[0]) % num_labels]
        noise_params = [noise_class, z0, z1]
        # number of elements in train dataset
        train_size = x_train.shape[0]
        print(model_name,
              "Labels for generated time series arrays: ",
              np.argmax(noise_class, axis=1))

        tv_plot = tv.train.PlotMetrics(columns=5, wait_num=5)
        for i in range(train_steps):
            # train the discriminator1 for 1 batch
            # 1 batch of real (label=1.0) and fake feature1 (label=0.0)
            # randomly pick real time series arrays from dataset
            dicta = {}
            rand_indexes = np.random.randint(0,
                                             train_size,
                                             size=batch_size)
            real_samples = x_train[rand_indexes]
            # real feature1 from encoder0 output
            real_feature0 = enc0.predict(real_samples)
            # generate random 50-dim z1 latent code
            real_z0 = np.random.normal(scale=0.5,
                                       size=[batch_size, z_dim])
            # real labels from dataset
            real_labels = y_train[rand_indexes]

            # generate fake feature1 using generator1 from
            # real labels and 50-dim z1 latent code
            fake_z0 = np.random.normal(scale=0.5,
                                       size=[batch_size, z_dim])
            fake_feature0 = gen0.predict([real_labels, fake_z0])

            # real + fake data
            feature0 = np.concatenate((real_feature0, fake_feature0))
            z0 = np.concatenate((fake_z0, fake_z0))

            # label 1st half as real and 2nd half as fake
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0

            # train discriminator1 to classify feature1 as
            # real/fake and recover
            # latent code (z0). real = from encoder1,
            # fake = from genenerator10
            # joint training using discriminator part of
            # advserial1 loss and entropy0 loss
            metrics = dis0.train_on_batch(feature0, [y, z0])
            # log the overall loss only
            log = "{}: [dis0_loss: {}]".format(i, metrics[0])
            dicta["dis0_loss"] = metrics[0]

            # train the discriminator1 for 1 batch
            # 1 batch of real (label=1.0) and fake time series arrays (label=0.0)
            # generate random 50-dim z1 latent code
            fake_z1 = np.random.normal(scale=0.5, size=[batch_size, z_dim])
            # generate fake time series arrays from real feature1 and fake z1
            fake_samples = gen1.predict([real_feature0, fake_z1])

            # real + fake data
            x = np.concatenate((real_samples, fake_samples))
            z1 = np.concatenate((fake_z1, fake_z1))

            # train discriminator1 to classify time series arrays
            # as real/fake and recover latent code (z1)
            # joint training using discriminator part of advserial0 loss
            # and entropy1 loss
            metrics = dis1.train_on_batch(x, [y, z1])
            # log the overall loss only (use dis1.metrics_names)
            log = "{} [dis1_loss: {}]".format(log, metrics[0])
            dicta["dis1_loss"] = metrics[0]

            # adversarial training
            # generate fake z0, labels
            fake_z0 = np.random.normal(scale=0.5,
                                       size=[batch_size, z_dim])
            # input to generator0 is sampling fr real labels and
            # 50-dim z0 latent code
            gen0_inputs = [real_labels, fake_z0]

            # label fake feature0 as real (specifies whether real or not)
            # is it bypassing the discriminator?
            y = np.ones([batch_size, 1])

            # train generator0 (thru adversarial) by fooling
            # the discriminator
            # and approximating encoder1 feature0 generator
            # joint training: adversarial0, entropy0, conditional0
            metrics = adv0.train_on_batch(gen0_inputs,
                                          [y, fake_z0, real_labels])
            fmt = "{} [adv0_loss: {}, enc1_acc: {}]"
            dicta["adv0_loss"] = metrics[0]
            dicta["enc1_acc"] = metrics[6]

            # log the overall loss and classification accuracy
            log = fmt.format(log, metrics[0], metrics[6])

            # input to generator0 is real feature0 and
            # 50-dim z0 latent code
            fake_z1 = np.random.normal(scale=0.5,
                                       size=[batch_size, z_dim])

            gen1_inputs = [real_feature0, fake_z1]

            # train generator1 (thru adversarial) by fooling
            # the discriminator and approximating encoder1 time series arrays
            # source generator joint training:
            # adversarial1, entropy1, conditional1
            metrics = adv1.train_on_batch(gen1_inputs,
                                          [y, fake_z1, real_feature0])
            # log the overall loss only
            log = "{} [adv1_loss: {}]".format(log, metrics[0])
            dicta["adv1_loss"] = metrics[0]

            print(log)
            if (i + 1) % save_interval == 0:
                generators = (gen0, gen1)
                self.plot_ts(generators,
                             noise_params=noise_params,
                             show=False,
                             step=(i + 1),
                             model_name=model_name)

            tv_plot.update({'dis0_loss': dicta["dis0_loss"], 'dis1_loss': dicta["dis1_loss"],
                            'adv0_loss': dicta["adv0_loss"], 'enc1_acc': dicta["enc1_acc"],
                            'adv1_loss': dicta["adv1_loss"]})
            # tv_plot.draw()

        # save the modelis after training generator0 & 1
        # the trained generator can be reloaded for
        # future data generation
        gen0.save(model_name + "-gen1.h5")
        gen1.save(model_name + "-gen0.h5")

        return gen0, gen1

    def plot_ts(self, generators,
                noise_params,
                show=False,
                step=0,
                model_name="gan"):
        """Generate fake time series arrays and plot them
        For visualization purposes, generate fake time series arrays
        then plot them in a square grid
        # Arguments
            generators (Models): gen0 and gen1 models for
                fake time series arrays generation
            noise_params (list): noise parameters
                (label, z0 and z1 codes)
            show (bool): Whether to show plot or not
            step (int): Appended to filename of the save time series arrays
            model_name (string): Model name
        """

        gen0, gen1 = generators
        noise_class, z0, z1 = noise_params
        feature0 = gen0.predict([noise_class, z0])
        tss = gen1.predict([feature0, z1])

    def train_encoder(self, model,
                      data,
                      model_name="MTSS-GAN"):
        """ Train the Encoder Model (enc0 and enc1)
        # Arguments
            model (Model): Encoder
            data (tensor): Train and test data
            model_name (string): model name
        """

        (x_train, y_train), (x_test, y_test) = data
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(x_train,
                  y_train,
                  validation_data=(x_test, y_test),
                  epochs=10,
                  batch_size=self.batch_size)

        model.save(model_name + "-encoder.h5")
        score = model.evaluate(x_test,
                               y_test,
                               batch_size=self.batch_size,
                               verbose=0)
        print("\nTest accuracy: {}%".format(100.0 * score[1]))

    def min_max(self, train_data):
        # Normalize the data.
        self.scaler = MinMaxScaler()
        num_instances, num_time_steps, num_features = train_data.shape
        train_data = np.reshape(train_data, (-1, num_features))
        # Train our scaler on the data.
        # TODO: Will we need the fit_transform().astype(np.float32) here?
        train_data = self.scaler.fit_transform(train_data)
        train_data = np.reshape(train_data, (num_instances, num_time_steps, num_features))
        return train_data

    def build_and_train_models(self, train_data, train_steps=2000):
        """Load the dataset, build MTSS discriminator,
        generator, and adversarial models.
        Call the MTSS train routine.
        """

        dataX, _, _ = google_data_loading(self.seq_len)
        train_data = end_cond(train_data)
        print('train_data.shape', train_data.shape)
        dataX = self.min_max(train_data=train_data)
        print('dataX.shape', dataX.shape)
        # dataX = np.dstack((dataX, train_data[:, :, -1]))
        print('dataX.shape', dataX.shape)
        dataX = np.stack(dataX)
        print('dataX.shape', dataX.shape)
        # for i in [0,1,2,3,20,21,50,300,1000,2000,3000,-1]:
        #     print(dataX[i][0])
        #     print(dataX[i][10])
        #     print(dataX[i][-10])
        #     print(dataX[i][-1])
        #     print('*'*50)

        train_n = int(len(dataX) * .70)
        X = dataX[:, :, :-1]
        y = dataX[:, -1, -1]
        print('X.shape', X.shape)
        print('y.shape', y.shape)
        x_train, y_train = X[:train_n, :, :], y[:train_n]
        x_test, y_test = X[train_n:, :, :], y[train_n:]
        print('x_train.shape', x_train.shape)
        print('x_test.shape', x_test.shape)
        print('y_train.shape', y_train.shape)
        print('y_test.shape', y_test.shape)

        # number of labels
        num_labels = len(np.unique(y_train))
        print('num_labels = ', num_labels)
        # to one-hot vector
        y_train = to_categorical(y_train, num_classes=num_labels)
        y_test = to_categorical(y_test, num_classes=num_labels)
        print('after to_categorical()')
        print('y_train.shape', y_train.shape)
        print('y_test.shape', y_test.shape)

        model_name = "MTSS-GAN"
        # network parameters
        # train_steps = 10
        # train_steps = 2000

        lr = 2e-4
        decay = 6e-8
        z_dim = 50  ##this is the real noise input
        z_shape = (z_dim,)
        feature0_dim = self.shape[0] * self.shape[1]
        feature0_shape = (feature0_dim,)
        # [1] uses Adam, but discriminator converges easily with RMSprop
        optimizer = RMSprop(lr=lr, decay=decay)

        # build discriminator 0 and Q network 0 models
        input_shape = (feature0_dim,)
        inputs = Input(shape=input_shape, name='discriminator0_input')
        dis0 = self.build_discriminator(inputs, z_dim=z_dim)
        # Model(Dense(self.shape[0]*self.shape[1]), [f0_source, z0_recon], name='dis0')

        # loss fuctions: 1) probability feature0 is real
        # (adversarial0 loss)
        # 2) MSE z0 recon loss (Q0 network loss or entropy0 loss)
        # Because there are two outputs.

        loss = ['binary_crossentropy', 'mse']
        loss_weights = [1.0, 1.0]
        dis0.compile(loss=loss,
                     loss_weights=loss_weights,
                     optimizer=optimizer,
                     metrics=['accuracy'])
        dis0.summary()  # feature0 discriminator, z0 estimator

        # build discriminator 1 and Q network 1 models

        input_shape = (x_train.shape[1], x_train.shape[2])
        inputs = Input(shape=input_shape, name='discriminator1_input')
        dis1 = self.discriminator(inputs, num_codes=z_dim)

        # loss fuctions: 1) probability time series arrays is real (adversarial1 loss)
        # 2) MSE z1 recon loss (Q1 network loss or entropy1 loss)
        loss = ['binary_crossentropy', 'mse']
        loss_weights = [1.0, 10.0]
        dis1.compile(loss=loss,
                     loss_weights=loss_weights,
                     optimizer=optimizer,
                     metrics=['accuracy'])
        dis1.summary()  # time series array discriminator, z1 estimator

        # build generator models
        label_shape = (num_labels,)
        feature0 = Input(shape=feature0_shape, name='feature0_input')
        labels = Input(shape=label_shape, name='labels')
        z0 = Input(shape=z_shape, name="z0_input")
        z1 = Input(shape=z_shape, name="z1_input")
        latent_codes = (labels, z0, z1, feature0)
        gen0, gen1 = self.build_generator(latent_codes)
        # gen0: classes and noise (labels + z0) to feature0
        gen0.summary()  # (latent features generator)
        # gen1: feature0 + z0 to feature1
        gen1.summary()  # (time series array generator )

        # build encoder models
        input_shape = self.shape
        inputs = Input(shape=input_shape, name='encoder_input')
        enc0, enc1 = self.build_encoder(inputs=(inputs, feature0),
                                        num_labels=num_labels)
        # Encoder0 or enc0: data to feature0
        enc0.summary()  # time series array to feature0 encoder
        # Encoder1 or enc1: feature0 to class labels
        enc1.summary()  # feature0 to labels encoder (classifier)
        encoder = Model(inputs, enc1(enc0(inputs)))
        encoder.summary()  # time series array to labels encoder (classifier)

        data = (x_train, y_train), (x_test, y_test)
        print('X.shape', X.shape)
        print('y.shape', y.shape)
        print('x_train.shape', x_train.shape)
        print('x_test.shape', x_test.shape)
        print('y_train.shape', y_train.shape)
        print('y_test.shape', y_test.shape)

        # this process would train enco, enc1, and encoder
        self.train_encoder(encoder, data, model_name=model_name)

        # build adversarial0 model =
        # generator0 + discriminator0 + encoder1
        # encoder0 weights frozen
        enc1.trainable = False
        # discriminator0 weights frozen
        dis0.trainable = False
        gen0_inputs = [labels, z0]
        gen0_outputs = gen0(gen0_inputs)
        adv0_outputs = dis0(gen0_outputs) + [enc1(gen0_outputs)]
        # labels + z0 to prob labels are real + z0 recon + feature1 recon
        adv0 = Model(gen0_inputs, adv0_outputs, name="adv0")
        # loss functions: 1) prob labels are real (adversarial1 loss)
        # 2) Q network 0 loss (entropy0 loss)
        # 3) conditional0 loss (classifier error)
        loss_weights = [1.0, 1.0, 1.0]
        loss = ['binary_crossentropy',
                'mse',
                'categorical_crossentropy']
        adv0.compile(loss=loss,
                     loss_weights=loss_weights,
                     optimizer=optimizer,
                     metrics=['accuracy'])
        adv0.summary()

        # build adversarial1 model =
        # generator1 + discriminator1 + encoder0
        optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
        # encoder1 weights frozen
        enc0.trainable = False
        # discriminator1 weights frozen
        dis1.trainable = False
        gen1_inputs = [feature0, z1]
        gen1_outputs = gen1(gen1_inputs)
        print(gen1_inputs)
        print(gen1_outputs)
        adv1_outputs = dis1(gen1_outputs) + [enc0(gen1_outputs)]
        # feature1 + z1 to prob feature1 is
        # real + z1 recon + feature1/time series array recon
        adv1 = Model(gen1_inputs, adv1_outputs, name="adv1")
        # loss functions: 1) prob feature1 is real (adversarial0 loss)
        # 2) Q network 1 loss (entropy1 loss)
        # 3) conditional1 loss
        loss = ['binary_crossentropy', 'mse', 'mse']
        loss_weights = [1.0, 10.0, 1.0]
        adv1.compile(loss=loss,
                     loss_weights=loss_weights,
                     optimizer=optimizer,
                     metrics=['accuracy'])
        adv1.summary()

        # train discriminator and adversarial networks
        models = (enc0, enc1, gen0, gen1, dis0, dis1, adv0, adv1)
        params = (self.batch_size, train_steps, num_labels, z_dim, model_name)
        gen0, gen1 = self.train(models, data, params)

        return gen0, gen1
    
    def train_gan(self, raw_data: np.ndarray, filename: str):
        # The raw data will be a numpy ndarray in the same format as the ori_data that we
        # load from the example data file, so will need to transform it and create the rolling
        # window sequences to train the GAN on.

        # We have to train the scaler on the data in the original format because we will need to
        # apply it to the raw data passed in to generate the next sequence.
        # Normalize Data
        # print('raw_data shape', raw_data.shape)
        # self.scaler = MinMaxScaler()
        # scaled_data = self.scaler.fit_transform(raw_data).astype(np.float32)
        # print('scaled_data shape', scaled_data.shape)

        # Create rolling window sequences
        data = []
        for i in range(len(raw_data) - self.seq_len):
            data.append(raw_data[i:i + self.seq_len])
        n_windows = len(data)
        print('len data = ', len(data))
        print('len data[0] = ', len(data[0]))

        # Perform the data manipulation that mtss-gan google data example does.
        train_data = np.stack(data)
        print('train_data.shape', train_data.shape)
        # train_data = np.dstack(())

        # TimeGAN Components
        # Network Parameters
        noise_dim = self.seq_len * self.n_seq
        hidden_dim = self.n_seq * 4

        self.gen0, self.gen1 = self.build_and_train_models(train_data=train_data,
                                                           train_steps=self.train_steps)

        # Save the GAN to disk
        self.save_gan(filename=filename)
        return

    def save_gan(self, filename: str):
        # Save the trained generator to disk.  We will need that later for imputing data.
        # Save model
        self.gen0.save(os.path.join(filename, 'gen0'))
        self.gen0.save_weights(os.path.join(filename, 'gen0', 'weights'))
        tf.keras.utils.plot_model(
            self.gen0, to_file=os.path.join(filename, 'gen0', 'model.png'), show_shapes=True,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
        self.gen1.save(os.path.join(filename, 'gen1'))
        self.gen1.save_weights(os.path.join(filename, 'gen1', 'weights'))
        tf.keras.utils.plot_model(
            self.gen1, to_file=os.path.join(filename, 'gen1', 'model.png'), show_shapes=True,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
        # Save normalizer
        scaler_filename = os.path.join(filename, "scaler.save")
        joblib.dump(self.scaler, scaler_filename)
        return

    def load_gan(self, filename: str):
        # Load the saved generator so we can generate data.
        # Load model
        self.gen0 = load_model(os.path.join(filename, 'gen0'))
        self.gen0.load_weights(os.path.join(filename, 'gen0', 'weights'))
        self.gen1 = load_model(os.path.join(filename, 'gen1'))
        self.gen1.load_weights(os.path.join(filename, 'gen1', 'weights'))
        # Load normalizer
        scaler_filename = os.path.join(filename, "scaler.save")
        self.scaler = joblib.load(scaler_filename)
        return
        
    def get_next_sequence(self, cur_sequence: np.ndarray) -> np.ndarray:
        # Given the current sequence of shape (seq_len,  n_seq) representing the currently
        # observed sequence, provide the next sequence of the same shape.
        #next_sequence = np.zeros((self.seq_len, self.n_seq))
        if self.use_random_z:
            # Use random data as input
            random_data = np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))
            Z_ = random_data[None,...]
        else:
            # Use given sequence as input (normalized)
            scaled_data = self.scaler.transform(cur_sequence).astype(np.float32)
            Z_ = scaled_data[None,...]
        # Generate synthetic sequence
        generated_data = [self.synthetic_data(Z_)] # list of only one window
        generated_data = np.array(np.vstack(generated_data))
        # Rescale
        generated_data = (self.scaler.inverse_transform(generated_data.reshape(-1, self.n_seq))
                          .reshape(-1, self.seq_len, self.n_seq))
        next_sequence = generated_data[0]
        return next_sequence

