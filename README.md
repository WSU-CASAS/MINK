# MINK

**M**issing data **I**mputation **N**ovel tool**K**it

MINK performs multiple data imputation for timestamped mobile sensor data.
Data are assumed to be sampled at 1Hz and all sensor readings are available for each available
timestamp.


#### Requirements
Use the *environment.mink.yml* file to build the python anaconda environment for MINK.

## How To Use MINK
There are 3 primary modes that MINK runs: **training**, **imputing**, and **evaluating**.
There is also a utility method for building the *ignorefile* which is used in **imputing** and **evaluating**.
For the following sections we will assume to have split our data into training and testing files.
We then create files from the testing data where we remove some data that needs to be imputed.

### Training
Many of the available methods train a model, if you select a method that doesn't require training the program will read in the training file but then quickly return.
For these examples we will use the *regression_rand_forest* method.
```
python mink --method=regression_rand_forest --trainingdata=data/file.train --spacing=0.1 --model-id=example.train
```

### Imputing and Evaluating
In order to properly evaluate our methods we need to know where gaps are in the ground truth data we will be comparing our imputed data to.
MINK uses the *ignorefile* to define where there are gaps that need to be ignored.
To create the *ignorefile*
```
python mink --method=regression_rand_forest --fulldata=data/file.test --buildignorefile
```
This creates the *ignorefile* at `data/file.test.ignore`

MINK can now just impute the data, impute the data and then evaluate the results, or evaluate the results of data you have already imputed.
Here we will impute and evaluate data that we have removed 10 percent of the data.
We need to include the model-id used when training the model above so MINK knows which one to use.
```
python mink --method=regression_rand_forest --imputedata=data/file.test.missing10 --fulldata=data/file.test --ignorefile=data/file.test.ignore --spacing=0.1 --model-id=example.train
```
This outputs the full file with imputed data at `data/file.test.missing10.regression_rand_forest.csv` and prints out the MAE values for each sensor.
If you wanted to only impute the data, then leave out the `--fulldata=data/file.test` argument.
If we wanted to then evaluate that imputed file, the command would look like this
```
python mink --method=regression_rand_forest --evaldata=data/file.test.missing10.regression_rand_forest.csv --fulldata=data/file.test --ignorefile=data/file.test.ignore --spacing=0.1 --model-id=example.train
```

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
