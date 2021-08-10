#!/usr/bin/python

# python mink.py data_file
#
# Performs multiple data imputation for timestamped mobile sensor data.
# Data are assumed to be sampled at 1Hz and all sensor readings are available
# for each available timestamp.

# Written by Brian L. Thomas and Diane J. Cook, Washington State University.

# Copyright (c) 2021. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import argparse
import collections
import copy
from datetime import datetime, timedelta
import os
import numpy as np
import time
import uuid
from mobiledata import MobileData
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import SimpleRNN, Dense


class PredictionObject:
    def __init__(self, num_past_events: int, index: int, model):
        # TODO: Add model_name
        # TODO: Add model_id
        # TODO: Add data_fields
        # TODO: Add model_directory
        # TODO: Add sensor_index_list
        # TODO: Add label_field_index
        self._num_past_events = num_past_events
        self.index = index
        self._model = model
        self.buffer = collections.deque()
        return

    def load_buffer(self, data: list):
        # Populate the sensor buffer.
        if len(self.buffer) < self._num_past_events:
            for i in range(self._num_past_events):
                self.buffer.append(data[i][self.index])
        return

    def add_reading(self, reading):
        self.buffer.popleft()
        self.buffer.append(reading[self.index])
        return

    def predict(self, stamp: datetime) -> float:
        vector = np.zeros((1, self._num_past_events+1))
        for i in range(self._num_past_events):
            vector[0][i] = self.buffer[i]
        vector[0][self._num_past_events] = float(stamp.hour)
        value = self._model.predict(vector)
        return value[0]


class PredictionDummy(PredictionObject):
    def __init__(self, num_past_events: int, index: int, model=None):
        super().__init__(num_past_events=num_past_events,
                         index=index,
                         model=model)
        return

    def load_buffer(self, data: list):
        return

    def add_reading(self, reading):
        return

    def predict(self, stamp: datetime) -> float:
        return None


class PredictionRegMLP(PredictionObject):
    def __init__(self, num_past_events: int, index: int, model=None):
        super().__init__(num_past_events=num_past_events,
                         index=index,
                         model=model)
        return

    def load_buffer(self, data: list):
        return

    def add_reading(self, reading):
        return

    def predict(self, stamp: datetime) -> float:
        return 0.0


class PredictionRegRandForest(PredictionObject):
    def __init__(self, num_past_events: int, index: int, model=None):
        super().__init__(num_past_events=num_past_events,
                         index=index,
                         model=model)
        return

    def load_buffer(self, data: list):
        return

    def add_reading(self, reading):
        return

    def predict(self, stamp: datetime) -> float:
        return 0.0


class PredictionRegSGD(PredictionObject):
    def __init__(self, num_past_events: int, index: int, model=None):
        super().__init__(num_past_events=num_past_events,
                         index=index,
                         model=model)
        return

    def load_buffer(self, data: list):
        return

    def add_reading(self, reading):
        return

    def predict(self, stamp: datetime) -> float:
        return 0.0


class PredictionGAN(PredictionObject):
    def __init__(self, num_past_events: int, index: int, model=None):
        super().__init__(num_past_events=num_past_events,
                         index=index,
                         model=model)
        return

    def load_buffer(self, data: list):
        return

    def add_reading(self, reading):
        return

    def predict(self, stamp: datetime) -> float:
        return 0.0


def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])


class MINK:
    def __init__(self):
        self.data_fields = collections.OrderedDict()
        self.num_sensors = 16
        self.event_spacing = timedelta(seconds=1.0)
        self._label_field = 'user_activity_label'
        self._label_field_index = -1
        self._has_label_field = False
        self._sensor_index_list = list()
        self._model_directory = 'models'
        self._num_past_events = 10
        self._overwrite_existing_models = False
        self._model_id = str(uuid.uuid4().hex)

        # Definitions
        self._gap_size = 10
        self._impute_field_mean = 'field_mean'
        self._impute_carry_forward = 'carry_forward'
        self._impute_carry_backward = 'carry_backward'
        self._impute_carry_average = 'carry_average'
        self._impute_regmlp = 'regression_mlp'
        self._impute_regrandforest = 'regression_rand_forest'
        self._impute_regsgd = 'regression_sgd'
        self._impute_regdnn = 'regression_dnn'
        self._impute_wavenet = 'wavenet'
        self._impute_gan = 'gan'
        self._impute_functions = dict({
            self._impute_field_mean: self._impute_func_field_mean,
            self._impute_carry_forward: self._impute_func_carry_forward,
            self._impute_carry_backward: self._impute_func_carry_backward,
            self._impute_carry_average: self._impute_func_carry_average,
            self._impute_regmlp: self._impute_func_regmlp,
            self._impute_regrandforest: self._impute_func_regrandforest,
            self._impute_regsgd: self._impute_func_regsgd,
            self._impute_regdnn: self._impute_func_regdnn,
            self._impute_wavenet: self._impute_func_wavenet,
            self._impute_gan: self._impute_func_gan})
        self._impute_methods = list(self._impute_functions.keys())
        self._impute_methods.sort()
        self._config_method = None
        self._config_datafile = None
        self._config_fulldatafile = None
        self.impute_func = self.impute_missing_values
        return

    @staticmethod
    def _get_run_logdir():
        logdir = os.path.join(os.curdir, 'logs', time.strftime('run_%Y_%m_%d-%H_%M_%S'))
        return logdir

    def read_data(self, datafile: str) -> list:
        """ Read and store sensor data from specified file. Assume that each reported
        time contains readings for all 16 sensors: yaw, pitch, roll, x/y/z rotation,
        x/y/z acceleration, latitude, longitude, altitude, course, speed,
        horizontal accuracy, and vertical accuracy.
        """
        with MobileData(file_path=datafile, mode='r') as in_data:
            self.data_fields = copy.deepcopy(in_data.fields)
            data = in_data.read_all_rows()

            # Count the number of sensors.
            self.num_sensors = 0
            i = 0
            # Create a list of just the sensor index positions so we don't have to check for more
            # than float vs string in most of our imputation methods.
            del self._sensor_index_list
            self._sensor_index_list = list()
            for field_name, field_type in self.data_fields.items():
                if field_name != self._label_field and field_type != 'dt':
                    self.num_sensors += 1
                    self._sensor_index_list.append(i)
                else:
                    self._has_label_field = True
                    self._label_field_index = i
                i += 1
        return data

    def report_data_statistics(self, data: list, segments: list):
        n = len(data)
        print('Number of available data readings:', n)
        print('Number of contiguous data segments:', len(segments))
        print('Number of data gaps:', len(segments) - 1)
        start_dt = data[0][0]
        end_dt = data[-1][0]
        dt = end_dt - start_dt
        num_seconds = dt.total_seconds()
        print('Data time span:  {}  ({} seconds)'.format(str(dt),
                                                         num_seconds))
        missing_delta = timedelta(seconds=0.0)
        for i in range(1, len(segments)):
            delta = segments[i]['first_stamp'] - segments[i - 1]['last_stamp']
            missing_delta += delta
        missing_seconds = missing_delta.total_seconds()
        print('Amount of missing time:  {}  ({} seconds)'.format(str(missing_delta),
                                                                 missing_seconds))
        print('Ratio of missing time:  {}%'
              .format(float(missing_seconds) / float(num_seconds) * 100.0))
        seg_start_stamp = data[segments[0]['first_index']][0]
        seg_end_stamp = data[segments[0]['last_index']][0]
        seg_size = segments[0]['last_index']
        self.event_spacing = abs(seg_end_stamp - seg_start_stamp) / float(seg_size)
        print('Extimated spacing between samples:  {}'.format(str(self.event_spacing)))
        print('\nData Segments:')
        for seg in segments:
            print(str(seg))
        return

    def impute_missing_values(self, data: list, segments: list) -> (list, datetime, list):
        newdata = list([self.num_sensors])
        dt = datetime.now()
        missing = list()

        return newdata, dt, missing

    def _get_data_segments(self, data: list) -> list:
        segments = list()

        # Find first data line with all sensors populated.
        field_types = list(self.data_fields.values())
        first_index = 0
        for i in range(len(data)):
            first_index = i
            is_full_values = True
            for j in range(len(data[i]) - 1):
                if data[i][j] is None and field_types[j] == 'f':
                    is_full_values = False
            if is_full_values:
                break

        print('segments, first_index={}'.format(first_index))

        # Initialize variables for our loop.
        new_segment = dict({'first_index': first_index,
                            'last_index': first_index,
                            'first_stamp': copy.deepcopy(data[first_index][0]),
                            'last_stamp': copy.deepcopy(data[-1][0])})
        one_second = timedelta(seconds=self._gap_size)
        for i in range(first_index, len(data)):
            # If the previous stamp is more than one second behind the current one, then we have
            # a gap that needs to be filled.  i-1 is the end of the previous segment and i is the
            # start of the current segment.
            if (data[i-1][0] + one_second) < data[i][0]:
                new_segment['last_index'] = i-1
                new_segment['last_stamp'] = copy.deepcopy(data[i-1][0])
                segments.append(copy.deepcopy(new_segment))
                del new_segment
                new_segment = dict({'first_index': i,
                                    'last_index': i,
                                    'first_stamp': copy.deepcopy(data[i][0]),
                                    'last_stamp': copy.deepcopy(data[i][0])})
        # We need to set the end of the final segment (even if it's the only segment) and add
        # it to the list.
        new_segment['last_index'] = len(data) - 1
        new_segment['last_stamp'] = copy.deepcopy(data[-1][0])
        segments.append(copy.deepcopy(new_segment))

        # Return the calculated segments.
        return segments

    def _populate_from_gap_values(self, data: list, segments: list,
                                  gap_values: list) -> (list, datetime, list):
        newdata = list()
        missing = list()

        # Start processing the data
        point_length = len(data[0])
        data_length = len(data)
        data_index = 0
        segment_index = 0
        gap_index = 0
        waiting_for_gap = True
        current_stamp = data[0][0]
        end_stamp = data[0][0]
        stamp_delta = copy.deepcopy(self.event_spacing)
        while current_stamp <= data[-1][0] and data_index < data_length:
            newpoint = list()
            if waiting_for_gap:
                # print('stamp = {}'.format(str(data[data_index][0])))
                newpoint.append(copy.deepcopy(data[data_index][0]))
                if data_index >= segments[segment_index]['last_index']:
                    if segment_index < (len(segments) - 1):
                        waiting_for_gap = False
                        segment_index += 1
                        current_stamp = copy.deepcopy(data[data_index][0]) + stamp_delta
                        end_stamp = copy.deepcopy(data[data_index + 1][0])
                for i in range(1, point_length):
                    newpoint.append(data[data_index][i])
                data_index += 1
                newdata.append(newpoint)
            else:
                # print('current_stamp={}'.format(str(current_stamp)))
                newpoint.append(copy.deepcopy(current_stamp))
                if current_stamp < end_stamp:
                    # print(str(data[data_index]))
                    # print(len(data[data_index]))
                    # print(str(gap_values[gap_index]))
                    # print(len(gap_values[gap_index]))
                    # print(point_length - 1)
                    for i in self._sensor_index_list:
                        newpoint.append(gap_values[gap_index][i - 1])
                    if self._has_label_field:
                        newpoint.append(None)
                    missing.append(len(newdata))
                    current_stamp += stamp_delta
                    newdata.append(newpoint)
                if current_stamp >= end_stamp:
                    waiting_for_gap = True
                    gap_index += 1

        dt = data[0][0]
        return newdata, dt, missing

    def _populate_from_models(self, data: list, segments: list,
                              model_list: list) -> (list, datetime, list):
        newdata = list()
        missing = list()

        # Start processing the data
        point_length = len(data[0])
        data_length = len(data)
        model_list_length = len(model_list)
        data_index = 0
        segment_index = 0
        gap_index = 0
        waiting_for_gap = True
        current_stamp = data[0][0]
        end_stamp = data[0][0]
        stamp_delta = copy.deepcopy(self.event_spacing)
        while current_stamp <= data[-1][0] and data_index < data_length:
            newpoint = list()
            if waiting_for_gap:
                newpoint.append(copy.deepcopy(data[data_index][0]))
                if data_index >= segments[segment_index]['last_index']:
                    if segment_index < (len(segments) - 1):
                        waiting_for_gap = False
                        segment_index += 1
                        current_stamp = copy.deepcopy(data[data_index][0]) + stamp_delta
                        end_stamp = copy.deepcopy(data[data_index + 1][0])
                for i in range(1, point_length):
                    newpoint.append(data[data_index][i])
                for i in range(model_list_length):
                    model_list[i].add_reading(reading=data[data_index])
                data_index += 1
                newdata.append(newpoint)
            else:
                newpoint.append(copy.deepcopy(current_stamp))
                if current_stamp < end_stamp:
                    for i in range(model_list_length):
                        newpoint.append(model_list[i].predict(stamp=current_stamp))
                    if self._has_label_field:
                        newpoint.append(None)
                    for i in range(model_list_length):
                        model_list[i].add_reading(reading=newpoint)
                    missing.append(len(newdata))
                    current_stamp += stamp_delta
                    newdata.append(newpoint)
                if current_stamp >= end_stamp:
                    waiting_for_gap = True
                    gap_index += 1

        dt = data[0][0]
        return newdata, dt, missing

    def _impute_func_field_mean(self, data: list, segments: list) -> (list, datetime, list):
        gap_values = list()
        # Build a list of the mean of each field
        means = list()
        for i, field_type in enumerate(self.data_fields.values()):
            if i in self._sensor_index_list:
                if field_type == 'f':
                    field = list()
                    for column in data:
                        if column[i] is not None:
                            field.append(column[i])
                    if len(field) == 0:
                        field.append(0)
                    means.append(np.mean(field))
                else:
                    means.append(None)
            else:
                means.append(None)

        # Build the gap values to use.
        for i in range(1, len(segments)):
            sensor_values = list()
            for s in self._sensor_index_list:
                sensor_values.append(means[s])
            gap_values.append(copy.deepcopy(sensor_values))
            del sensor_values

        newdata, dt, missing = self._populate_from_gap_values(data=data,
                                                              segments=segments,
                                                              gap_values=gap_values)
        return newdata, dt, missing

    def _impute_func_carry_forward(self, data: list, segments: list) -> (list, datetime, list):
        gap_values = list()
        for i in range(1, len(segments)):
            sensor_values = list()
            for s in self._sensor_index_list:
                value = data[segments[i - 1]['last_index']][s]
                sensor_values.append(value)
            gap_values.append(copy.deepcopy(sensor_values))
            del sensor_values

        newdata, dt, missing = self._populate_from_gap_values(data=data,
                                                              segments=segments,
                                                              gap_values=gap_values)
        return newdata, dt, missing

    def _impute_func_carry_backward(self, data: list, segments: list) -> (list, datetime, list):
        gap_values = list()
        for i in range(1, len(segments)):
            sensor_values = list()
            for s in self._sensor_index_list:
                value = data[segments[i]['first_index']][s]
                sensor_values.append(value)
            gap_values.append(copy.deepcopy(sensor_values))
            del sensor_values

        newdata, dt, missing = self._populate_from_gap_values(data=data,
                                                              segments=segments,
                                                              gap_values=gap_values)
        return newdata, dt, missing

    def _impute_func_carry_average(self, data: list, segments: list) -> (list, datetime, list):
        gap_values = list()
        for i in range(1, len(segments)):
            sensor_values = list()
            for s, field_type in enumerate(self.data_fields.values()):
                if field_type == 'dt':
                    continue
                if field_type == 'f' and s in self._sensor_index_list:
                    value = None
                    if data[segments[i - 1]['last_index']][s] is not None \
                            and data[segments[i]['first_index']][s] is not None:
                        value = data[segments[i - 1]['last_index']][s] \
                                + data[segments[i]['first_index']][s]
                        value = float(value) / 2.0
                    sensor_values.append(value)
                else:
                    sensor_values.append(None)
            gap_values.append(copy.deepcopy(sensor_values))
            del sensor_values

        newdata, dt, missing = self._populate_from_gap_values(data=data,
                                                              segments=segments,
                                                              gap_values=gap_values)
        return newdata, dt, missing

    def _impute_func_regmlp(self, data: list, segments: list) -> (list, datetime, list):
        self._make_model_directory(directory=self._model_directory)
        model_list = list()

        for s, field_type in enumerate(self.data_fields.values()):
            if s not in self._sensor_index_list:
                continue
            if field_type == 'f':
                model_name = 'MLP.{}.{}.model'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                train_model = True

                if not self._overwrite_existing_models and os.path.exists(model_filename):
                    train_model = False

                if train_model:
                    vector, target = self._build_sensor_feature_vector(data=data,
                                                                       segments=segments,
                                                                       index=s)

                    print('Training model: {}'.format(model_name))
                    model = MLPRegressor().fit(vector, target)
                    joblib.dump(value=model,
                                filename=model_filename)

        for s, field_type in enumerate(self.data_fields.values()):
            if field_type == 'dt' or s == self._label_field_index:
                continue
            if field_type == 'f':
                model_name = 'MLP.{}.{}.model'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                model = joblib.load(model_filename)
                model_list.append(PredictionObject(num_past_events=self._num_past_events,
                                                   index=s,
                                                   model=model))
            else:
                model_list.append(PredictionDummy(num_past_events=self._num_past_events,
                                                  index=s))

        # Prime the buffers.
        for i in range(len(model_list)):
            model_list[i].load_buffer(data=data)

        newdata, dt, missing = self._populate_from_models(data=data,
                                                          segments=segments,
                                                          model_list=model_list)
        return newdata, dt, missing

    def _impute_func_regrandforest(self, data: list, segments: list) -> (list, datetime, list):
        self._make_model_directory(directory=self._model_directory)
        model_list = list()

        for s, field_type in enumerate(self.data_fields.values()):
            if s not in self._sensor_index_list:
                continue
            if field_type == 'f':
                model_name = 'RandForest.{}.{}.model'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                train_model = True

                if not self._overwrite_existing_models and os.path.exists(model_filename):
                    train_model = False

                if train_model:
                    vector, target = self._build_sensor_feature_vector(data=data,
                                                                       segments=segments,
                                                                       index=s)

                    print('Training model: {}'.format(model_name))
                    model = RandomForestRegressor(max_depth=20,
                                                  min_samples_split=5,
                                                  n_jobs=30)
                    model.fit(vector, target)
                    joblib.dump(value=model,
                                filename=model_filename)

        for s, field_type in enumerate(self.data_fields.values()):
            if field_type == 'dt' or s == self._label_field_index:
                continue
            if field_type == 'f':
                model_name = 'RandForest.{}.{}.model'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                model = joblib.load(model_filename)
                model_list.append(PredictionObject(num_past_events=self._num_past_events,
                                                   index=s,
                                                   model=model))
            else:
                model_list.append(PredictionDummy(num_past_events=self._num_past_events,
                                                  index=s))

        # Prime the buffers.
        for i in range(len(model_list)):
            model_list[i].load_buffer(data=data)

        newdata, dt, missing = self._populate_from_models(data=data,
                                                          segments=segments,
                                                          model_list=model_list)
        return newdata, dt, missing

    def _impute_func_regsgd(self, data: list, segments: list) -> (list, datetime, list):
        self._make_model_directory(directory=self._model_directory)
        model_list = list()

        for s, field_type in enumerate(self.data_fields.values()):
            if s not in self._sensor_index_list:
                continue
            if field_type == 'f':
                model_name = 'SGD.{}.{}.model'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                train_model = True

                if not self._overwrite_existing_models and os.path.exists(model_filename):
                    train_model = False

                if train_model:
                    vector, target = self._build_sensor_feature_vector(data=data,
                                                                       segments=segments,
                                                                       index=s)

                    print('Training model: {}'.format(model_name))
                    model = make_pipeline(StandardScaler(),
                                          SGDRegressor())
                    model.fit(vector, target)
                    joblib.dump(value=model,
                                filename=model_filename)

        for s, field_type in enumerate(self.data_fields.values()):
            if field_type == 'dt' or s == self._label_field_index:
                continue
            if field_type == 'f':
                model_name = 'SGD.{}.{}.model'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                model = joblib.load(model_filename)
                model_list.append(PredictionObject(num_past_events=self._num_past_events,
                                                   index=s,
                                                   model=model))
            else:
                model_list.append(PredictionDummy(num_past_events=self._num_past_events,
                                                  index=s))

        # Prime the buffers.
        for i in range(len(model_list)):
            model_list[i].load_buffer(data=data)

        newdata, dt, missing = self._populate_from_models(data=data,
                                                          segments=segments,
                                                          model_list=model_list)
        return newdata, dt, missing

    def _impute_func_regdnn(self, data: list, segments: list) -> (list, datetime, list):
        self._make_model_directory(directory=self._model_directory)
        newdata = data
        dt = datetime.now()
        missing = list([self.num_sensors])

        return newdata, dt, missing

    def _impute_func_wavenet(self, data: list, segments: list) -> (list, datetime, list):
        self._make_model_directory(directory=self._model_directory)
        model_list = list()

        for s, field_type in enumerate(self.data_fields.values()):
            if s not in self._sensor_index_list:
                continue
            if field_type == 'f':
                model_name = 'WaveNet.{}.{}.model'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                train_model = True

                if not self._overwrite_existing_models and os.path.exists(model_filename):
                    train_model = False

                if train_model:
                    vector = self._build_sensor_feature_vector_i(data=data,
                                                                 segments=segments,
                                                                 index=s)

                    print(vector.shape)
                    dataset_size = len(vector)
                    train_size = int(dataset_size * 0.7)
                    valid_size = train_size + int(dataset_size * 0.2)

                    x_train = vector[:train_size, :self._num_past_events]
                    y_train = vector[:train_size, -1]
                    x_valid = vector[train_size:valid_size, :self._num_past_events]
                    y_valid = vector[train_size:valid_size, -1]
                    x_test = vector[valid_size:, :self._num_past_events]
                    y_test = vector[valid_size:, -1]

                    print('training size = {}'.format(train_size))
                    print(x_train.shape)
                    print(y_train.shape)
                    print('validation size = {}'.format(valid_size - train_size))
                    print(x_valid.shape)
                    print(y_valid.shape)
                    print('testing size = {}'.format(dataset_size - valid_size))
                    print(x_test.shape)
                    print(y_test.shape)

                    print('Training model: {}'.format(model_name))
                    length = self._num_past_events
                    n_features = len(vector[0])

                    model = keras.models.Sequential()
                    model.add(keras.layers.InputLayer(input_shape=[None, 1]))
                    for rate in (1, 2, 4, 8) * 2:
                        model.add(keras.layers.Conv1D(filters=20,
                                                      kernel_size=2,
                                                      padding="causal",
                                                      activation="relu",
                                                      dilation_rate=rate))
                    model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
                    optimizer = keras.optimizers.Adam(learning_rate=0.002)
                    model.compile(loss="mse",
                                  optimizer=optimizer,
                                  metrics=[last_time_step_mse,
                                           keras.metrics.MeanSquaredError(),
                                           keras.metrics.MeanAbsoluteError(),
                                           keras.metrics.KLDivergence()])
                    run_logdir = self._get_run_logdir()
                    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
                    history = model.fit(x_train, y_train,
                                        epochs=10,
                                        validation_data=(x_valid, y_valid),
                                        callbacks=[tensorboard_cb])

                    # Save the model to disk.
                    # keras.models.save_model(model=model,
                    #                         filepath=model_filename)
                    model.save(filepath=model_filename)

                    del vector
                    del model
                    # model = keras.models.load_model(
                    #     model_filename,
                    #     custom_objects={'last_time_step_mse': last_time_step_mse})
                    # del model

        for s, field_type in enumerate(self.data_fields.values()):
            if field_type == 'dt' or s == self._label_field_index:
                continue
            if field_type == 'f':
                model_name = 'WaveNet.{}.{}.model'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                model = keras.models.load_model(
                    model_filename,
                    custom_objects={'last_time_step_mse': last_time_step_mse})
                model_list.append(PredictionObject(num_past_events=self._num_past_events,
                                                   index=s,
                                                   model=model))
            else:
                model_list.append(PredictionDummy(num_past_events=self._num_past_events,
                                                  index=s))

        # Prime the buffers.
        for i in range(len(model_list)):
            model_list[i].load_buffer(data=data)

        newdata, dt, missing = self._populate_from_models(data=data,
                                                          segments=segments,
                                                          model_list=model_list)

        return newdata, dt, missing

    def _impute_func_gan(self, data: list, segments: list) -> (list, datetime, list):
        self._make_model_directory(directory=self._model_directory)
        model_list = list()

        for s, field_type in enumerate(self.data_fields.values()):
            if s not in self._sensor_index_list:
                continue
            if field_type == 'f':
                model_name = 'GAN.{}.{}.model.hdf5'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                train_model = True

                if not self._overwrite_existing_models and os.path.exists(model_filename):
                    train_model = False

                if train_model:
                    vector = self._build_sensor_feature_vector_i(data=data,
                                                                 segments=segments,
                                                                 index=s)

                    print(vector.shape)
                    dataset_size = len(vector)
                    train_size = int(dataset_size * 0.7)
                    valid_size = train_size + int(dataset_size * 0.2)

                    x_train = vector[:train_size, :self._num_past_events]
                    y_train = vector[:train_size, -1]
                    x_valid = vector[train_size:valid_size, :self._num_past_events]
                    y_valid = vector[train_size:valid_size, -1]
                    x_test = vector[valid_size:, :self._num_past_events]
                    y_test = vector[valid_size:, -1]

                    # print(vector[train_size])
                    print('training size = {}'.format(train_size))
                    print(x_train.shape)
                    print(y_train.shape)
                    print('validation size = {}'.format(valid_size - train_size))
                    print(x_valid.shape)
                    print(y_valid.shape)
                    print('testing size = {}'.format(dataset_size - valid_size))
                    print(x_test.shape)
                    print(y_test.shape)

                    print('Training model: {}'.format(model_name))
                    length = self._num_past_events
                    n_features = len(vector[0])
                    """
                    generator = TimeseriesGenerator(data=vector,
                                                    targets=target,
                                                    stride=3,
                                                    length=length,
                                                    batch_size=128)
                    model = Sequential()
                    model.add(SimpleRNN(100, activation='relu', input_shape=(length, n_features)))
                    model.add(Dense(100, activation='relu', input_shape=(length, n_features)))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss="mse")
                    model.fit(generator)

                    keras.models.save_model(model=model,
                                            filepath=model_filename)
                    """
                    model = keras.models.Sequential()
                    model.add(keras.layers.InputLayer(input_shape=[None, 1]))
                    for rate in (1, 2, 4, 8) * 2:
                        model.add(keras.layers.Conv1D(filters=20,
                                                      kernel_size=2,
                                                      padding="causal",
                                                      activation="relu",
                                                      dilation_rate=rate))
                    model.add(keras.layers.Conv1D(filters=1, kernel_size=1))
                    model.compile(loss="mse",
                                  optimizer="adam",
                                  metrics=[last_time_step_mse])
                    history = model.fit(x_train, y_train,
                                        epochs=20,
                                        validation_data=(x_valid, y_valid))

                    del vector
                    del model

        for s, field_type in enumerate(self.data_fields.values()):
            if field_type == 'dt' or s == self._label_field_index:
                continue
            if field_type == 'f':
                model_name = 'GAN.{}.{}.model.h5'.format(self._model_id, s)
                model_filename = os.path.join(self._model_directory, model_name)
                model = keras.models.load_model(model_filename)
                model_list.append(PredictionObject(num_past_events=self._num_past_events,
                                                   index=s,
                                                   model=model))
            else:
                model_list.append(PredictionDummy(num_past_events=self._num_past_events,
                                                  index=s))

        # Prime the buffers.
        for i in range(len(model_list)):
            model_list[i].load_buffer(data=data)

        newdata, dt, missing = self._populate_from_models(data=data,
                                                          segments=segments,
                                                          model_list=model_list)

        return newdata, dt, missing

    @staticmethod
    def _make_model_directory(directory: str):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        return

    def _build_sensor_feature_vector(self, data: list, segments: list, index: int):
        vector = None
        target = None

        for i in range(len(segments)):
            s_start = segments[i]['first_index']
            s_end = segments[i]['last_index'] + 1
            np_v, np_t = self._build_feature_vector(data=data[s_start:s_end],
                                                    index=index)
            if np_v is not None and np_t is not None:
                if vector is None:
                    vector = np_v
                else:
                    vector = np.vstack((vector, np_v))
                if target is None:
                    target = np_t
                else:
                    target = np.hstack((target, np_t))
            print(vector.shape)
        return vector, target

    def _build_feature_vector(self, data: list, index: int):
        length = len(data) - self._num_past_events
        width = self._num_past_events + 1

        # Check to see if we have enough data to build a feature vector.
        if length < 1:
            return None, None
        vector = np.zeros((length, width))
        target = np.zeros(length)
        buffer = collections.deque()

        # Populate the sensor buffer.
        for i in range(self._num_past_events):
            buffer.append(data[i][index])

        for i in range(self._num_past_events, len(data)):
            for j in range(self._num_past_events):
                vector[i - self._num_past_events][j] = buffer[j]
            vector[i - self._num_past_events][width - 1] = float(data[i][0].hour)
            target[i - self._num_past_events] = data[i][index]
            buffer.popleft()
            buffer.append(data[i][index])
        return vector, target

    def _build_sensor_feature_vector_i(self, data: list, segments: list, index: int):
        vector = None

        for i in range(len(segments)):
            s_start = segments[i]['first_index']
            s_end = segments[i]['last_index'] + 1
            np_v = self._build_feature_vector_i(data=data[s_start:s_end],
                                                index=index)
            if np_v is not None:
                if vector is None:
                    vector = np_v
                else:
                    vector = np.vstack((vector, np_v))
        return vector

    def _build_feature_vector_i(self, data: list, index: int):
        length = len(data) - self._num_past_events
        width = self._num_past_events + 1

        # Check to see if we have enough data to build a feature vector.
        if length < 1:
            return None, None
        vector = np.zeros((length, width, 1))
        buffer = collections.deque()

        # Populate the sensor buffer.
        for i in range(width):
            buffer.append(data[i][index])

        for i in range(width, len(data)):
            for j in range(width):
                vector[i - self._num_past_events][j][0] = buffer[j]
            buffer.popleft()
            buffer.append(data[i][index])
        return vector

    def normalize_longitude(self):
        # -180 to 180
        return

    def un_normalize_longitude(self):
        return

    def normalize_latitude(self):
        # -90 to 90
        return

    def un_normalize_latitude(self):
        return

    def report_data(self, filename: str, data: list):
        # Create the filename that we want to write to.
        out_filename = '{}.complete'.format(filename)

        # Instruct the data layer to write out our imputed data.
        MobileData.write_rows_to_file(file_name=out_filename,
                                      fields=self.data_fields,
                                      rows_to_write=data,
                                      mode='w')
        return

    def evaluate(self, fulldata: list, imputted_data: list, missing: list):
        min_values = np.zeros(len(self.data_fields))
        max_values = np.zeros(len(self.data_fields))
        mae_values = np.zeros(len(self.data_fields))
        denominators = np.zeros(len(self.data_fields))
        num_missing = len(missing)
        for i, field_type in enumerate(self.data_fields.values()):
            if field_type == 'f':
                field = list()
                for column in fulldata:
                    if column[i] is not None:
                        field.append((column[i]))
                if len(field) == 0:
                    field.append(0)
                min_values[i] = np.min(field)
                max_values[i] = np.max(field)
                denominators[i] = max_values[i] - min_values[i]
        for item in missing:
            for i, field_type in enumerate(self.data_fields.values()):
                if field_type == 'f':
                    if fulldata[item][i] is not None and imputted_data[item][i] is not None:
                        mae_values[i] += np.abs(fulldata[item][i] - imputted_data[item][i])
        for i, field_type in enumerate(self.data_fields.values()):
            if field_type == 'f':
                mae_values[i] /= num_missing
        for i, field_type in enumerate(self.data_fields.values()):
            if field_type == 'f':
                if denominators[i] != 0.0:
                    mae_values[i] = (mae_values[i] - min_values[i]) / denominators[i]
        if num_missing == 0:
            print("No values are missing")
        else:
            print("Normalized MAE:", np.sum(mae_values) / float(self.num_sensors))
            for i, field_name in enumerate(self.data_fields.keys()):
                print('{} MAE: {}'.format(field_name, mae_values[i]))
        return

    def run(self):
        # Run the command line parser.
        self.run_config()

        # Now we can do the work!
        # Read in the data file.
        data = self.read_data(datafile=self._config_datafile)
        # Get the complete data segments.
        segments = self._get_data_segments(data=data)
        # Print out a quick analysis of the initial data statistics.
        self.report_data_statistics(data=data,
                                    segments=segments)
        # Run the imputation function.
        newdata, start, missing = self.impute_func(data=data,
                                                   segments=segments)
        # Write out the imputed data.
        self.report_data(filename=self._config_datafile,
                         data=newdata)

        if self._config_fulldatafile is not None:
            fulldata = self.read_data(datafile=self._config_fulldatafile)
            self.evaluate(fulldata=fulldata,
                          imputted_data=newdata,
                          missing=missing)
        return

    def run_config(self):
        parser = argparse.ArgumentParser(description='Missing data Imputation Novel toolKit.')
        parser.add_argument('--method',
                            dest='method',
                            choices=self._impute_methods,
                            required=True,
                            help='Choose the method to use when imputing the missing data values.')
        parser.add_argument('--data',
                            dest='data',
                            type=str,
                            required=True,
                            help='The data file with missing data to impute.')
        parser.add_argument('--fulldata',
                            dest='fulldata',
                            type=str,
                            help=('The complete data file to run calculations against the '
                                  'imputed data.'))
        parser.add_argument('--gapsize',
                            dest='gapsize',
                            type=int,
                            default=self._gap_size,
                            help='The minimum number of seconds to be considered a gap to impute.')
        parser.add_argument('--model-id',
                            dest='model_id',
                            type=str,
                            required=False,
                            default=self._model_id,
                            help=('An ID you can assign to your models to identify them '
                                  'separately from other models'))
        args = parser.parse_args()

        self._model_id = args.model_id
        self._gap_size = args.gapsize
        self._config_method = args.method
        self._config_datafile = args.data
        self._config_fulldatafile = args.fulldata
        self.impute_func = self._impute_functions[self._config_method]
        return


if __name__ == "__main__":
    worker = MINK()
    worker.run()
