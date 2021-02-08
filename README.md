# MINK

**M**issing data **I**mputation **N**ovel **T**oolkit

MINK performs multiple data imputation for timestamped mobile sensor data.
Data are assumed to be sampled at 1Hz and all sensor readings are available for each available
timestamp.


#### Requirements
*numpy*


## Running
To run MINK and generate a completed data file you only need to pass a data file.
```
python mink.py data
```

To run the full evaluation and calculate the normalized MAE provide the original full data as a
second argument.
