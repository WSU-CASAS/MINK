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
import sys
from datetime import datetime, timedelta

import numpy as np


class MINK:
    def __init__(self):
        self.num_sensors = 16
        self._impute_methods = ['field_mean',
                                'carry_forward',
                                'carry_backward',
                                'carry_average']
        self._config_method = None
        self._config_datafile = None
        self._config_fulldatafile = None
        return

    @staticmethod
    def get_datetime(date: str, time: str) -> datetime:
        """ Input is two strings representing a date and a time with the format
        YYYY-MM-DD HH:MM:SS.ms. This function converts the two strings to a single
        datetime.datetime() object.
        """
        stamp = date + ' ' + time
        if '.' in stamp:  # Remove optional millisecond precision
            stamp = stamp.split('.', 1)[0]
        dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
        return dt

    @staticmethod
    def read_entry(infile):
        """ Parse a single line from a text file containing a sensor reading.
        The format is "date time sensorname sensorname value <activitylabel|0>".
        """
        try:
            line = infile.readline()
            x = str(str(line).strip()).split(' ', 5)
            if len(x) < 6:
                return True, x[0], x[1], x[2], x[3], x[4], 'None'
            else:
                x[5] = x[5].replace(' ', '_')
                return True, x[0], x[1], x[2], x[3], x[4], x[5]
        except:
            return False, None, None, None, None, None, None

    def read_data(self, datafile: str) -> list:
        """ Read and store sensor data from specified file. Assume that each reported
        time contains readings for all 16 sensors: yaw, pitch, roll, x/y/z rotation,
        x/y/z acceleration, latitude, longitude, altitude, course, speed,
        horizontal accuracy, and vertical accuracy.
        """
        infile = open(datafile, "r")
        valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
        count = 0
        data = list()
        valid = True
        while valid:
            datapoint = []
            dt = self.get_datetime(date, time)
            datapoint.append(dt)
            while count < 16:
                datapoint.append(float(v1))
                valid, date, time, f1, f2, v1, v2 = self.read_entry(infile)
                count += 1
            data.append(datapoint)
            count = 0
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
        means = np.zeros((self.num_sensors + 1))
        missing = []
        for i in range(1, self.num_sensors):
            field = [column[i] for column in data]
            means[i] = np.mean(field)
        newdata = []
        start = data[0][0]
        index = 0
        while len(newdata) <= num_seconds:
            newpoint = []
            if start < data[index][0]:
                for i in range(1, self.num_sensors + 1):
                    newpoint.append(means[i])
                    if i == 1:
                        missing.append(len(newdata))
            else:
                for i in range(1, self.num_sensors + 1):
                    newpoint.append(data[index][i])
                index += 1
            newdata.append(newpoint)
            start += timedelta(0, 1)
        return newdata, data[0][0], missing

    @staticmethod
    def report_data(filename: str, data: list, start: datetime):
        outfile = open(filename + ".complete", "w")
        n = len(data)
        for i in range(n):
            newtime = start + timedelta(0, i)
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
        newdata, start, missing = self.impute_missing_values(data=data,
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
        return


if __name__ == "__main__":
    worker = MINK()
    worker.run()
