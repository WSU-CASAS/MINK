#!/usr/bin/python

# python mink.py data_file
#
# Performs multiple data imputation for timestamped mobile sensor data.
# Data are assumed to be sampled at 1Hz and all sensor readings are available
# for each available timestamp.

# Written by Brian Thomas and Diane J. Cook, Washington State University.

# Copyright (c) 2021. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import sys
from datetime import datetime, timedelta

import numpy as np

num_sensors = 16
data = []


def get_datetime(date, time):
    """ Input is two strings representing a date and a time with the format
    YYYY-MM-DD HH:MM:SS.ms. This function converts the two strings to a single
    datetime.datetime() object.
    """
    stamp = date + ' ' + time
    if '.' in stamp:  # Remove optional millsecond precision
        stamp = stamp.split('.', 1)[0]
    dt = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")
    return dt


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


def read_data(datafile):
    """ Read and store sensor data from specified file. Assume that each reported
    time contains readings for all 16 sensors: yaw, pitch, roll, x/y/z rotation,
    x/y/z acceleration, latitude, longitude, altitude, course, speed,
    horizontal accuracy, and vertical accuracy.
    """
    infile = open(datafile, "r")
    valid, date, time, f1, f2, v1, v2 = read_entry(infile)
    count = 0
    valid = True
    while valid:
        datapoint = []
        dt = get_datetime(date, time)
        datapoint.append(dt)
        while count < 16:
            datapoint.append(float(v1))
            valid, date, time, f1, f2, v1, v2 = read_entry(infile)
            count += 1
        data.append(datapoint)
        count = 0
    return data


def report_data_statistics(data):
    n = len(data)
    print("Number of available data readings:", n)
    start = data[0][0]
    end = data[-1][0]
    dt = end - start
    num_seconds = dt.total_seconds()
    print("Data time span:", dt, "(", num_seconds, "seconds )")
    num_missing = num_seconds - n
    print("Number of missing values:", num_missing, "(",
          float(num_missing) / float(num_seconds), "% )")
    return int(num_seconds)


def impute_missing_values(data, num_seconds):
    means = np.zeros((num_sensors + 1))
    missing = []
    for i in range(1, num_sensors):
        field = [column[i] for column in data]
        means[i] = np.mean(field)
    newdata = []
    start = data[0][0]
    index = 0
    while len(newdata) <= num_seconds:
        newpoint = []
        if start < data[index][0]:
            for i in range(1, num_sensors + 1):
                newpoint.append(means[i])
                if i == 1:
                    missing.append(len(newdata))
        else:
            for i in range(1, num_sensors + 1):
                newpoint.append(data[index][i])
            index += 1
        newdata.append(newpoint)
        start += timedelta(0, 1)
    return newdata, data[0][0], missing


def report_data(filename, data, start):
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


def evaluate(fulldata, imputted_data, missing):
    min_values = np.zeros(num_sensors)
    max_values = np.zeros(num_sensors)
    mae_values = np.zeros(num_sensors)
    denominators = np.zeros(num_sensors)
    num_missing = len(missing)
    for i in range(num_sensors):
        field = [column[i + 1] for column in fulldata]  # ignore field 0 of fulldata
        min_values[i] = np.min(field)
        max_values[i] = np.max(field)
        denominators[i] = max_values[i] - min_values[i]
    for item in missing:
        for i in range(num_sensors):
            mae_values[i] += np.abs(fulldata[item][i + 1] - imputted_data[item][i])
    for i in range(num_sensors):
        mae_values[i] /= num_missing
    for i in range(num_sensors):
        if denominators[i] != 0.0:
            mae_values[i] = (mae_values[i] - min_values[i]) / denominators[i]
    if num_missing == 0:
        print("No values are missing")
    else:
        print("Normalized MAE:", np.sum(mae_values) / float(num_sensors))


if __name__ == "__main__":
    data = read_data(sys.argv[1])
    num_seconds = report_data_statistics(data)
    newdata, start, missing = impute_missing_values(data, num_seconds)
    report_data(sys.argv[1], newdata, start)
    if len(sys.argv) > 2:
        fulldata = read_data(sys.argv[2])
        evaluate(fulldata, newdata, missing)
