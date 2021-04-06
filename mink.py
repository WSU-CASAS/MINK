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

import numpy as np
from mobiledata import MobileData


class MINK:
    def __init__(self):
        self.data_fields = collections.OrderedDict()
        self.num_sensors = 16
        self.event_spacing = timedelta(seconds=1.0)
        self._impute_field_mean = 'field_mean'
        self._impute_carry_forward = 'carry_forward'
        self._impute_carry_backward = 'carry_backward'
        self._impute_carry_average = 'carry_average'
        self._impute_functions = dict({
            self._impute_field_mean: self._impute_func_field_mean,
            self._impute_carry_forward: self._impute_func_carry_forward,
            self._impute_carry_backward: self._impute_func_carry_backward,
            self._impute_carry_average: self._impute_func_carry_average})
        self._impute_methods = list(self._impute_functions.keys())
        self._impute_methods.sort()
        self._config_method = None
        self._config_datafile = None
        self._config_fulldatafile = None
        self.impute_func = self.impute_missing_values
        return

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
            for field_name, field_type in self.data_fields.items():
                if field_type == 'f':
                    self.num_sensors += 1

            # Estimate the sample spacing.
            if len(data) > 11:
                start_stamp = data[0][0]
                end_stamp = data[10][0]
                self.event_spacing = abs(end_stamp - start_stamp) / 10.0
        return data

    @staticmethod
    def report_data_statistics(data: list) -> int:
        n = len(data)
        print("Number of available data readings:", n)
        start_dt = data[0][0]
        end_dt = data[-1][0]
        dt = end_dt - start_dt
        num_seconds = dt.total_seconds()
        print("Data time span:", dt, "(", num_seconds, "seconds )")
        num_missing = num_seconds - n
        print("Number of missing values:", num_missing, "(",
              float(num_missing) / float(num_seconds), "% )")
        return int(num_seconds)

    def impute_missing_values(self, data: list, num_seconds: int) -> (list, datetime, list):
        newdata = list([self.num_sensors])
        dt = datetime.now()
        missing = list()

        return newdata, dt, missing

    @staticmethod
    def _get_data_segments(data: list) -> list:
        segments = list()

        # Initialize variables for our loop.
        new_segment = dict({'first_index': 0,
                            'last_index': 0,
                            'first_stamp': copy.deepcopy(data[0][0]),
                            'last_stamp': copy.deepcopy(data[-1][0])})
        one_second = timedelta(seconds=1)
        for i in range(1, len(data)):
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

    def _impute_func_field_mean(self, data: list, num_seconds: int) -> (list, datetime, list):
        # print('first = {}'.format(str(data[0][0])))
        # print('last  = {}'.format(str(data[-1][0])))
        newdata = list()
        missing = list()

        # Build a numpy array of the mean of each field
        means = np.zeros((self.num_sensors + 1))
        for i in range(1, self.num_sensors):
            field = [column[i] for column in data]
            means[i] = np.mean(field)

        # Start processing the data
        start = data[0][0]
        index = 0
        while len(newdata) <= num_seconds:
            newpoint = list()
            # print('index = {}  start = {}  data[index][0] = {}'.format(index,
            #                                                            str(start),
            #                                                            str(data[index][0])))
            if start < data[index][0]:
                for i in range(1, self.num_sensors + 1):
                    newpoint.append(means[i])
                    if i == 1:
                        missing.append(len(newdata))
                # print('newpoint = {}'.format(str(newpoint)))
                # print('missing = {}'.format(str(missing)))
            else:
                for i in range(1, self.num_sensors + 1):
                    newpoint.append(data[index][i])
                index += 1
                # print('newpoint = {}'.format(str(newpoint)))
            newdata.append(newpoint)
            start += timedelta(0, 1)
        dt = data[0][0]
        return newdata, dt, missing

    def _populate_from_gap_values(self, data: list, num_seconds: int, segments: list,
                                  gap_values: list) -> (list, datetime, list):
        newdata = list()
        missing = list()

        # Start processing the data
        data_index = 0
        segment_index = 0
        gap_index = 0
        waiting_for_gap = True
        current_stamp = data[0][0]
        end_stamp = data[0][0]
        stamp_delta = copy.deepcopy(self.event_spacing)
        while len(newdata) <= num_seconds:
            # print('dind={}  sind={}  gind={}  wait={}  curstm={}  end_stm={}'
            #       .format(data_index,
            #               segment_index,
            #               gap_index,
            #               waiting_for_gap,
            #               current_stamp,
            #               end_stamp))
            newpoint = list()
            if waiting_for_gap:
                if data_index >= segments[segment_index]['last_index']:
                    if segment_index < (len(segments) - 1):
                        waiting_for_gap = False
                        segment_index += 1
                        current_stamp = copy.deepcopy(data[data_index][0]) + stamp_delta
                        end_stamp = copy.deepcopy(data[data_index + 1][0])
                for i in range(1, self.num_sensors + 1):
                    newpoint.append(data[data_index][i])
                data_index += 1
            else:
                if current_stamp < end_stamp:
                    for i in range(self.num_sensors):
                        newpoint.append(gap_values[gap_index][i])
                    missing.append(len(newdata))
                    current_stamp += stamp_delta
                if current_stamp >= end_stamp:
                    waiting_for_gap = True
                    gap_index += 1
            newdata.append(newpoint)
        dt = data[0][0]
        return newdata, dt, missing

    def _impute_func_carry_forward(self, data: list, num_seconds: int) -> (list, datetime, list):
        segments = self._get_data_segments(data=data)
        print('Number of data segments: {}'.format(len(segments)))
        for seg in segments:
            print(str(seg))
        gap_values = list()
        for i in range(1, len(segments)):
            sensor_values = list()
            for s in range(1, self.num_sensors + 1):
                value = data[segments[i - 1]['last_index']][s]
                sensor_values.append(value)
            gap_values.append(copy.deepcopy(sensor_values))
            del sensor_values

        newdata, dt, missing = self._populate_from_gap_values(data=data,
                                                              num_seconds=num_seconds,
                                                              segments=segments,
                                                              gap_values=gap_values)
        return newdata, dt, missing

    def _impute_func_carry_backward(self, data: list, num_seconds: int) -> (list, datetime, list):
        segments = self._get_data_segments(data=data)
        gap_values = list()
        for i in range(1, len(segments)):
            sensor_values = list()
            for s in range(1, self.num_sensors + 1):
                value = data[segments[i]['first_index']][s]
                sensor_values.append(value)
            gap_values.append(copy.deepcopy(sensor_values))
            del sensor_values

        newdata, dt, missing = self._populate_from_gap_values(data=data,
                                                              num_seconds=num_seconds,
                                                              segments=segments,
                                                              gap_values=gap_values)
        return newdata, dt, missing

    def _impute_func_carry_average(self, data: list, num_seconds: int) -> (list, datetime, list):
        segments = self._get_data_segments(data=data)
        gap_values = list()
        for i in range(1, len(segments)):
            sensor_values = list()
            for s in range(1, self.num_sensors + 1):
                value = data[segments[i - 1]['last_index']][s] \
                        + data[segments[i]['first_index']][s]
                value = float(value) / 2.0
                sensor_values.append(value)
            gap_values.append(copy.deepcopy(sensor_values))
            del sensor_values

        newdata, dt, missing = self._populate_from_gap_values(data=data,
                                                              num_seconds=num_seconds,
                                                              segments=segments,
                                                              gap_values=gap_values)
        return newdata, dt, missing

    @staticmethod
    def report_data(filename: str, data: list, start: datetime):
        outfile = open(filename + ".complete", "w")
        n = len(data)
        for i in range(n):
            newtime = start + timedelta(0, i)
            # print('{}  {}'.format(str(newtime), str(data[i])))
            dtstr = newtime.strftime("%Y-%m-%d %H:%M:%S")
            outstr = dtstr + " Yaw Yaw " + str(data[i][0]) + " 0\n"
            outstr += dtstr + " Pitch Pitch " + str(data[i][1]) + " 0\n"
            outstr += dtstr + " Roll Roll " + str(data[i][2]) + " 0\n"
            outstr += dtstr + " RotationRateX RotationRateX " + str(data[i][3]) + " 0\n"
            outstr += dtstr + " RotationRateY RotationRateY " + str(data[i][4]) + " 0\n"
            outstr += dtstr + " RotationRateZ RotationRateZ " + str(data[i][5]) + " 0\n"
            outstr += dtstr + " UserAccelerationX UserAccelerationX " + str(data[i][6]) + " 0\n"
            outstr += dtstr + " UserAccelerationY UserAccelerationY " + str(data[i][7]) + " 0\n"
            outstr += dtstr + " UserAccelerationZ UserAccelerationZ " + str(data[i][8]) + " 0\n"
            outstr += dtstr + " Latitude Latitude " + str(data[i][9]) + " 0\n"
            outstr += dtstr + " Longitude Longitude " + str(data[i][10]) + " 0\n"
            outstr += dtstr + " Altitude Altitude " + str(data[i][11]) + " 0\n"
            outstr += dtstr + " Course Course " + str(data[i][12]) + " 0\n"
            outstr += dtstr + " Speed Speed " + str(data[i][13]) + " 0\n"
            outstr += dtstr + " HorizontalAccuracy HorizontalAccuracy " + str(data[i][14]) + " 0\n"
            outstr += dtstr + " VerticalAccuracy VerticalAccuracy " + str(data[i][15]) + " 0\n"
            outfile.write(outstr)
        outfile.close()
        return

    def evaluate(self, fulldata: list, imputted_data: list, missing: list):
        min_values = np.zeros(self.num_sensors)
        max_values = np.zeros(self.num_sensors)
        mae_values = np.zeros(self.num_sensors)
        denominators = np.zeros(self.num_sensors)
        num_missing = len(missing)
        for i in range(self.num_sensors):
            field = [column[i + 1] for column in fulldata]  # ignore field 0 of fulldata
            min_values[i] = np.min(field)
            max_values[i] = np.max(field)
            denominators[i] = max_values[i] - min_values[i]
        for item in missing:
            for i in range(self.num_sensors):
                mae_values[i] += np.abs(fulldata[item][i + 1] - imputted_data[item][i])
        for i in range(self.num_sensors):
            mae_values[i] /= num_missing
        for i in range(self.num_sensors):
            if denominators[i] != 0.0:
                mae_values[i] = (mae_values[i] - min_values[i]) / denominators[i]
        if num_missing == 0:
            print("No values are missing")
        else:
            print("Normalized MAE:", np.sum(mae_values) / float(self.num_sensors))
        return

    def run(self):
        # Run the command line parser.
        self.run_config()

        # Now we can do the work.
        data = self.read_data(datafile=self._config_datafile)
        num_seconds = self.report_data_statistics(data=data)
        newdata, start, missing = self.impute_func(data=data,
                                                   num_seconds=num_seconds)
        self.report_data(filename=self._config_datafile,
                         data=newdata,
                         start=start)

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
        args = parser.parse_args()

        self._config_method = args.method
        self._config_datafile = args.data
        self._config_fulldatafile = args.fulldata
        self.impute_func = self._impute_functions[self._config_method]
        return


if __name__ == "__main__":
    worker = MINK()
    worker.run()
