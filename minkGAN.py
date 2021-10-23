import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam


class MinkGAN:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.batch_size = 128
        self.train_steps = 100 # Set to 10000 for production
        self.columns = list(['yaw', 'pitch', 'roll', 'rotation_rate_x', 'rotation_rate_y',
                             'rotation_rate_z', 'user_acceleration_x', 'user_acceleration_y',
                             'user_acceleration_z', 'latitude', 'longitude', 'altitude', 'course',
                             'speed', 'horizontal_accuracy', 'vertical_accuracy'])
        self.n_seq = len(self.columns)
        self.scaler = None
        self.synthetic_data = None # will hold synthetic data generator model
        self.use_random_z = True # use random (True) or previous sequence (False) as input to model
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
    
    def train_gan(self, raw_data: np.ndarray, filename: str):
        # The raw data will be a numpy ndarray in the same format as the ori_data that we
        # load from the example data file, so will need to transform it, create the rolling
        # window sequences, and tf.data.Dataset object to train the GAN on.

        # Normalize Data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(raw_data).astype(np.float32)
        
        # Create rolling window sequences
        data = []
        for i in range(len(raw_data) - self.seq_len):
            data.append(scaled_data[i:i + self.seq_len])
        n_windows = len(data)

        # Create tf.data.Dataset
        real_series = (tf.data.Dataset
                       .from_tensor_slices(data)
                       .shuffle(buffer_size=n_windows)
                       .batch(self.batch_size))
        real_series_iter = iter(real_series.repeat())
        
        random_series = iter(tf.data.Dataset
                             .from_generator(self._make_random_data, output_types=tf.float32)
                             .batch(self.batch_size)
                             .repeat())
        
        # TimeGAN Components
        # Network Parameters
        hidden_dim = 24
        num_layers = 3
        # Input place holders
        X = Input(shape=[self.seq_len, self.n_seq], name='RealData')
        Z = Input(shape=[self.seq_len, self.n_seq], name='RandomData')
        
        # Embedder and Recovery
        embedder = self._make_rnn(n_layers=3,
                                  hidden_units=hidden_dim,
                                  output_units=hidden_dim,
                                  name='Embedder')
        recovery = self._make_rnn(n_layers=3,
                                  hidden_units=hidden_dim,
                                  output_units=self.n_seq,
                                  name='Recovery')
        # Generator and Discriminator
        generator = self._make_rnn(n_layers=3,
                                   hidden_units=hidden_dim,
                                   output_units=hidden_dim,
                                   name='Generator')
        discriminator = self._make_rnn(n_layers=3,
                                       hidden_units=hidden_dim,
                                       output_units=1,
                                       name='Discriminator')
        supervisor = self._make_rnn(n_layers=2,
                                    hidden_units=hidden_dim,
                                    output_units=hidden_dim,
                                    name='Supervisor')

        # TimeGAN Training
        # Settings
        gamma = 1
        # Generic Loss Functions
        mse = MeanSquaredError()
        bce = BinaryCrossentropy()
        
        # Phase 1 - Autoencoder Training
        H = embedder(X)
        X_tilde = recovery(H)

        autoencoder = Model(inputs=X,
                            outputs=X_tilde,
                            name='Autoencoder')
        # Autoencoder Optimizer
        autoencoder_optimizer = Adam()
        # Autoencoder training step
        
        @tf.function
        def train_autoencoder_init(x):
            with tf.GradientTape() as tape:
                x_tilde = autoencoder(x)
                embedding_loss_t0 = mse(x, x_tilde)
                e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)
            var_list = embedder.trainable_variables + recovery.trainable_variables
            gradients = tape.gradient(e_loss_0, var_list)
            autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
            return tf.sqrt(embedding_loss_t0)
        
        # Autoencoder training loop
        for step in tqdm(range(self.train_steps)):
            X_ = next(real_series_iter)
            step_e_loss_t0 = train_autoencoder_init(X_)
        
        # Phase 2 - Supervised Training
        # Define optimizer
        supervisor_optimizer = Adam()
        # Train Step
        
        @tf.function
        def train_supervisor(x):
            with tf.GradientTape() as tape:
                h = embedder(x)
                h_hat_supervised = supervisor(h)
                g_loss_s = mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])
            var_list = supervisor.trainable_variables
            gradients = tape.gradient(g_loss_s, var_list)
            supervisor_optimizer.apply_gradients(zip(gradients, var_list))
            return g_loss_s
        
        # Training loop
        for step in tqdm(range(self.train_steps)):
            X_ = next(real_series_iter)
            step_g_loss_s = train_supervisor(X_)
        
        # Joint Training
        # Generator
        # Adversarial Architecture - Supervised
        E_hat = generator(Z)
        H_hat = supervisor(E_hat)
        Y_fake = discriminator(H_hat)
        
        adversarial_supervised = Model(inputs=Z,
                                       outputs=Y_fake,
                                       name='AdversarialNetSupervised')
        # Adversarial Architecture in Latent Space
        Y_fake_e = discriminator(E_hat)
        
        adversarial_emb = Model(inputs=Z,
                                outputs=Y_fake_e,
                                name='AdversarialNet')
        # Mean and Variance loss
        X_hat = recovery(H_hat)
        self.synthetic_data = Model(inputs=Z,
                                    outputs=X_hat,
                                    name='SyntheticData')
        
        def get_generator_moment_loss(y_true, y_pred):
            y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
            y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
            g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
            g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
            return g_loss_mean + g_loss_var

        # Discriminator
        # Architecture - Real Data
        Y_real = discriminator(H)
        discriminator_model = Model(inputs=X,
                                    outputs=Y_real,
                                    name='DiscriminatorReal')
        # Optimizers
        generator_optimizer = Adam()
        discriminator_optimizer = Adam()
        embedding_optimizer = Adam()
        # Generator Train Step
        
        @tf.function
        def train_generator(x, z):
            with tf.GradientTape() as tape:
                y_fake = adversarial_supervised(z)
                generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake),
                                                  y_pred=y_fake)
                y_fake_e = adversarial_emb(z)
                generator_loss_unsupervised_e = bce(y_true=tf.ones_like(y_fake_e),
                                                    y_pred=y_fake_e)
                h = embedder(x)
                h_hat_supervised = supervisor(h)
                generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])
        
                x_hat = self.synthetic_data(z)
                generator_moment_loss = get_generator_moment_loss(x, x_hat)
        
                generator_loss = (generator_loss_unsupervised +
                                  generator_loss_unsupervised_e +
                                  100 * tf.sqrt(generator_loss_supervised) +
                                  100 * generator_moment_loss)
        
            var_list = generator.trainable_variables + supervisor.trainable_variables
            gradients = tape.gradient(generator_loss, var_list)
            generator_optimizer.apply_gradients(zip(gradients, var_list))
            return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss
        
        # Embedding Train Step

        @tf.function
        def train_embedder(x):
            with tf.GradientTape() as tape:
                h = embedder(x)
                h_hat_supervised = supervisor(h)
                generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])
                x_tilde = autoencoder(x)
                embedding_loss_t0 = mse(x, x_tilde)
                e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised
        
            var_list = embedder.trainable_variables + recovery.trainable_variables
            gradients = tape.gradient(e_loss, var_list)
            embedding_optimizer.apply_gradients(zip(gradients, var_list))
            return tf.sqrt(embedding_loss_t0)
        
        # Discriminator Train Step

        @tf.function
        def get_discriminator_loss(x, z):
            y_real = discriminator_model(x)
            discriminator_loss_real = bce(y_true=tf.ones_like(y_real),
                                          y_pred=y_real)
            y_fake = adversarial_supervised(z)
            discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake),
                                          y_pred=y_fake)
            y_fake_e = adversarial_emb(z)
            discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e),
                                            y_pred=y_fake_e)
            return (discriminator_loss_real +
                    discriminator_loss_fake +
                    gamma * discriminator_loss_fake_e)
        
        @tf.function
        def train_discriminator(x, z):
            with tf.GradientTape() as tape:
                discriminator_loss = get_discriminator_loss(x, z)
        
            var_list = discriminator.trainable_variables
            gradients = tape.gradient(discriminator_loss, var_list)
            discriminator_optimizer.apply_gradients(zip(gradients, var_list))
            return discriminator_loss
        
        # Training Loop
        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
        for step in range(self.train_steps):
            # Train generator (twice as often as discriminator)
            for kk in range(2):
                X_ = next(real_series_iter)
                Z_ = next(random_series)
        
                # Train generator
                step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_, Z_)
                # Train embedder
                step_e_loss_t0 = train_embedder(X_)
        
            X_ = next(real_series_iter)
            Z_ = next(random_series)
            step_d_loss = get_discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                step_d_loss = train_discriminator(X_, Z_)
        
            if step % 1000 == 0:
                print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                      f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | '
                      f'e_loss_t0: {step_e_loss_t0:6.4f}')

        # Save the GAN to disk
        self.save_gan(filename=filename)
        return

    def save_gan(self, filename: str):
        # Save the trained generator to disk.  We will need that later for imputing data.
        # Save model
        self.synthetic_data.save(filename)
        # Save normalizer
        scaler_filename = os.path.join(filename, "scaler.save")
        joblib.dump(self.scaler, scaler_filename)
        return

    def load_gan(self, filename: str):
        # Load the saved generator so we can generate data.
        # Load model
        self.synthetic_data = tf.keras.models.load_model(filename)
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
            Z_ = [np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))]
            print(Z_)
        else:
            # Use given sequence as input (normalized)
            scaled_data = self.scalar.transform(cur_sequence).astype(np.float32)
            Z_ = tf.data.Dataset.from_tensors(scaled_data)
        # Generate synthetic sequence
        generated_data = [self.synthetic_data(Z_)] # list of only one window
        generated_data = np.array(np.vstack(generated_data))
        # Rescale
        generated_data = (self.scaler.inverse_transform(generated_data.reshape(-1, self.n_seq))
                          .reshape(-1, self.seq_len, self.n_seq))
        next_sequence = generated_data[0]
        return next_sequence

