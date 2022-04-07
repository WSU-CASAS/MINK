# MINK

**M**issing data **I**mputation **N**ovel tool**K**it

MINK performs multiple data imputation for timestamped mobile sensor data.
Data are assumed to be sampled at 1Hz and all sensor readings are available for each available
timestamp.


#### Requirements
Use the *environment.mink.yml* file to build the python anaconda environment for MINK.

## Usage
```
usage: mink.py [-h] --method {carry_average,carry_backward,carry_forward,field_mean,gan,regression_dnn,regression_mlp,regression_rand_forest,regression_sgd,wavenet,zero_fill} [--trainingdata TRAININGDATA] [--imputedata IMPUTEDATA] [--evaldata EVALDATA] [--fulldata FULLDATA] [--ignorefile IGNOREFILE]
               [--spacing SPACING] [--gapsize GAPSIZE] [--seqlen SEQLEN] [--model-id MODEL_ID] [--buildignorefile]

Missing data Imputation Novel toolKit.

optional arguments:
  -h, --help            show this help message and exit
  --method {carry_average,carry_backward,carry_forward,field_mean,gan,regression_dnn,regression_mlp,regression_rand_forest,regression_sgd,wavenet,zero_fill}
                        Choose the method to use when imputing the missing data values.
  --trainingdata TRAININGDATA
                        The data that will be used to train our models.
  --imputedata IMPUTEDATA
                        The data file with missing data to impute.
  --evaldata EVALDATA   The imputed data file to evaluation with fulldata.
  --fulldata FULLDATA   The complete data file to run calculations against the imputed data.
  --ignorefile IGNOREFILE
                        The file containing the datetime windows that will be ignored during imputation and evaluation. This file can be automatically generated if you use --buildignorefile.
  --spacing SPACING     The spacing between samples in seconds.
  --gapsize GAPSIZE     The minimum number of seconds to be considered a gap to impute.
  --seqlen SEQLEN       The length of the training sequence length. default=100.
  --model-id MODEL_ID   An ID you can assign to your models to identify them separately from other models
  --buildignorefile     Instructs the program to automatically generate an ignore file saved to the filename provided to --ignorefile using the data provided to --fulldata. When used the program will generate the ignore file and then exit.
```
