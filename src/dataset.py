import os
import numpy as np
import pandas as pd
import keras


class NaskdaqDataGenerator(keras.utils.Sequence):
    def __init__(self, data_generate_obj, batch_size, shuffle=True):
        self.data_generate_obj = data_generate_obj
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(self.data_generate_obj.sample_length / self.batch_size))

    def __getitem__(self, index):
        x_train, y_train = self.data_generate_obj.train_next_batch(index, self.batch_size)
        return x_train, y_train
        # print(x_train.shape)

    def on_epoch_end(self):
        self.data_generate_obj.shuffle_data_indexes()


class TimeSeriesNaskdaq(object):
    def __init__(self, input_length, output_length, n_dim):
        self.file_path = "/home/wuming/repo/WGAN-TensorFlow/data/nasdaq100_padding.csv"
        # self.file_path = "/Users/wuming/CodeRepo/dl/WGAN-TensorFlow/data/nasdaq100_padding.csv"
        self.input_length = input_length
        self.output_length = output_length
        self.n_dim = n_dim
        self.step = 3
        self.x_trains = None
        self.y_trains = None
        self.indexes = list()

    def __preprare_ts_data__(self):
        data_frame = pd.read_csv(self.file_path)
        index_names = data_frame.columns
        sub_index_names = index_names[:self.n_dim]
        sub_data_frame = data_frame[sub_index_names]
        normal_sub_data_frame = (sub_data_frame - sub_data_frame.mean()) / sub_data_frame.std()
        print(normal_sub_data_frame)
        return normal_sub_data_frame.values

    @property
    def sample_length(self):
        return len(self.x_trains)

    def load_data(self):
        normal_data_frame = self.__preprare_ts_data__()
        x_trains, y_trains = list(), list()
        for st in range(0, len(normal_data_frame) - self.input_length - self.output_length + 1, self.step):
            sub_x_frame = normal_data_frame[st:st + self.input_length]
            sub_y_frame = normal_data_frame[st + self.input_length:st + self.input_length + self.output_length]
            x_trains.append(sub_x_frame)
            y_trains.append(sub_y_frame)
        self.x_trains = np.array(x_trains)
        self.y_trains = np.array(y_trains)
        self.indexes = np.arange(len(self.x_trains))

    def shuffle_data_indexes(self):
        np.random.shuffle(self.indexes)

    def train_next_batch(self, index, batch_size):
        select_idx = self.indexes[index*batch_size:(index+1)*batch_size]
        return self.x_trains[select_idx], self.y_trains[select_idx]
