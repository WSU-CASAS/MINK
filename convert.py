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


with MobileData('data/sttr001.test', 'r') as in_data, \
        MobileData('data/sttr001.test.simple', 'w') as out_data:
    # You could use in_data.fields here to get the original headers, and filter for just 'f' ones
    out_fields = collections.OrderedDict({
        'yaw': 'f',
        'pitch': 'f',
        'roll': 'f',
        'rotation_rate_x': 'f',
        'rotation_rate_y': 'f',
        'rotation_rate_z': 'f',
        'user_acceleration_x': 'f',
        'user_acceleration_y': 'f',
        'user_acceleration_z': 'f',
        'latitude': 'f',
        'longitude': 'f',
        'altitude': 'f',
        'course': 'f',
        'speed': 'f',
        'horizontal_accuracy': 'f',
        'vertical_accuracy': 'f'})
    out_data.set_fields(out_fields)

    out_data.write_headers()
    # each 'row' is a dict with key->value for each input field
    for row in in_data.rows_dict:
        for key in out_fields.keys():
            if row[key] is None:
                row[key] = 0.0
        out_data.write_row_dict(row)
        # I think that the out_data object will only use the fields you specified above,
        # even though more were set in the row dict.
