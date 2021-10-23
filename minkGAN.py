import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam


class MinkGAN:
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.columns = list(['yaw', 'pitch', 'roll', 'rotation_rate_x', 'rotation_rate_y',
                             'rotation_rate_z', 'user_acceleration_x', 'user_acceleration_y',
                             'user_acceleration_z', 'latitude', 'longitude', 'altitude', 'course',
                             'speed', 'horizontal_accuracy', 'vertical_accuracy'])
        self.n_seq = len(self.columns)
        return

    def train_gan(self, raw_data: np.ndarray, filename: str):
        # The raw data will be a numpy ndarray in the same format as the ori_data that we
        # load from the example data file, so will need to transform it, create the rolling
        # window sequences, and tf.data.Dataset object to train the GAN on.

        # Do work here.

        # Save the GAN to disk at the end.
        self.save_gan(filename=filename)
        return

    def save_gan(self, filename: str):
        # Save the trained generator to disk.  We will need that later for imputing data.
        return

    def load_gan(self, filename: str):
        # Load the saved generator so we can generate data.
        return

    def get_next_sequence(self, cur_sequence: np.ndarray) -> np.ndarray:
        # Given the current sequence of shape (seq_len,  n_seq) representing the currently
        # observed sequence, provide the next sequence of the same shape.
        next_sequence = np.zeros((self.seq_len, self.n_seq))
        return next_sequence

