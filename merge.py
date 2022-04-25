import argparse
import collections
import copy
from datetime import datetime, timedelta
import os
import numpy as np
import time
import uuid
from mobiledata import MobileData
from typing import OrderedDict


def read_data(datafile: str) -> (list, OrderedDict):
    """ Read and store sensor data from specified file. Assume that each reported
    time contains readings for all 16 sensors: yaw, pitch, roll, x/y/z rotation,
    x/y/z acceleration, latitude, longitude, altitude, course, speed,
    horizontal accuracy, and vertical accuracy.
    """
    with MobileData(file_path=datafile, mode='r') as in_data:
        all_data_fields = copy.deepcopy(in_data.fields)
        all_data = in_data.read_all_rows()
    return all_data, all_data_fields


def write_data(filename: str, out_data: list, out_data_fields: OrderedDict):
    # Instruct the data layer to write out our merged imputed data.
    MobileData.write_rows_to_file(file_name=filename,
                                  fields=out_data_fields,
                                  rows_to_write=out_data,
                                  mode='w')
    return


methods = list(['carry_forward', 'field_mean', 'ms_gan'])
files = list(['sttr001', 'sttr002', 'sttr003', 'sttr004', 'sttr006', 'sttr007', 'sttr008',
              'sttr009', 'sttr010', 'sttr011', 'sttr012', 'sttr013', 'sttr014', 'sttr015',
              'sttr016', 'sttr101', 'sttr102', 'sttr103', 'sttr104', 'sttr105', 'sttr106',
              'sttr107', 'sttr108', 'sttr109', 'sttr110', 'sttr111', 'sttr112', 'sttr113',
              'sttr114'])
output_files = list()
for i in range(len(files)):
    files[i] = '../../full-data/{}.sampled.csv'.format(files[i])
    output_files.append('{}.OUTPUT.imputed.csv'.format(files[i]))
type_files = dict()
for method in methods:
    type_files[method] = list()
    for i in range(len(files)):
        type_files[method].append('{}.{}.imputed.csv'.format(files[i], method))

sensor_methods = dict({
    # 'yaw': 'gan',
    # 'pitch': 'gan',
    # 'roll': 'gan',
    # 'rotation_rate_x': 'gan',
    # 'rotation_rate_y': 'gan',
    # 'rotation_rate_z': 'gan',
    # 'user_acceleration_x': 'gan',
    # 'user_acceleration_y': 'gan',
    # 'user_acceleration_z': 'gan',
    'latitude': 'carry_forward',
    'longitude': 'carry_forward',
    'altitude': 'carry_forward',
    'course': 'carry_forward',
    'speed': 'carry_forward',
    'horizontal_accuracy': 'field_mean',
    'vertical_accuracy': 'field_mean',
    'battery_state': 'carry_forward'
})
data_fields = None

for file_index in range(len(files)):
    print('{}\tworking {}'.format(str(datetime.now()), files[file_index]))
    data = dict()
    # Read in the data files for each method type.
    for method in methods:
        print('\t{}\treading {}'.format(str(datetime.now()), type_files[method][file_index]))
        data[method], data_fields = read_data(type_files[method][file_index])
    # Merge the values into a single data list, we will use the GAN as the base.
    data['out'] = data['ms_gan']
    print('\t{}\tmerging data...'.format(str(datetime.now())))
    for i in range(len(data['out'])):
        for j, field_name in enumerate(data_fields.values()):
            if field_name in sensor_methods:
                # Copy over the data from the defined sensor method.
                data['out'][i][j] = data[sensor_methods[field_name]][i][j]

    # Write out the data to the file.
    print('\t{}\twriting file {}'.format(str(datetime.now()), output_files[file_index]))
    write_data(filename=output_files[file_index],
               out_data=data['out'],
               out_data_fields=data_fields)


