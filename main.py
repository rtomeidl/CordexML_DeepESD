####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
import warnings; warnings.simplefilter("ignore", category=Warning)
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

####################################################################################################################################################################################
####################################################################################################################################################################################
from sklearn.model_selection import train_test_split

import xarray as xr
import numpy as np

import json
import sys
import gc

####################################################################################################################################################################################
####################################################################################################################################################################################
from src.utils import *
from src.cnn import *

####################################################################################################################################################################################
####################################################################################################################################################################################
# # # def MyWho():
# # #  print([v for v in globals().keys() if not v.startswith('_')])

####################################################################################################################################################################################
####################################################################################################################################################################################
print('READ CONFIGURATION')
config_file  = json.load(open('./config/config.json'))
threshold_pr = config_file['threshold_pr']
output_path  = config_file['output_path']
nbatch_size  = config_file['nbatch_size']
patience_es  = config_file['patience_es']
input_path   = config_file['input_path']
target_var   = config_file['target_var']
train_exp    = config_file['train_exp']
ml_model     = config_file['ml_model']
nepochs      = config_file['nepochs']
domain       = config_file['domain']
lossf        = config_file['lossf']
lrate        = config_file['lrate']
del config_file, json
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
if (target_var != 'tasmax') and (target_var != 'pr'):
  sys.exit(' ... Unkown target variable')

if (lossf != 'customBGamma') and (lossf != 'customMSE') and (lossf != 'SqrMSE'):
  sys.exit(' ... Unkown loss function')

####################################################################################################################################################################################
####################################################################################################################################################################################
if ((lossf != 'SqrMSE') and (lossf != 'customBGamma')) and (target_var == 'pr'):
  sys.exit(' ... Pr is only compatible with SqrMSE or customBGamma')

if (lossf != 'customMSE') and (target_var == 'tasmax'):
  sys.exit(' ... Tasmax is only compatible with customMSE')

####################################################################################################################################################################################
####################################################################################################################################################################################
print('LOAD PREDICTORS')
predictor = load_data(input_path, train_exp, domain, 'predictors')

####################################################################################################################################################################################
####################################################################################################################################################################################
print('LOAD PREDICTANDS')
predictand = load_data(input_path, train_exp, domain, 'target', target_var)
del input_path, load_data
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
print('REMOVE NANS')
predictor  = remove_days_with_nans(predictor, {'lat':'lat', 'lon':'lon'})

if   domain == 'ALPS_domain' : predictand = remove_days_with_nans(predictand,{'lat':'y'  , 'lon':'x'  })
elif domain == 'NZ_domain'   : predictand = remove_days_with_nans(predictand,{'lat':'lat', 'lon':'lon'})
elif domain == 'SA_domain'   : predictand = remove_days_with_nans(predictand,{'lat':'lat', 'lon':'lon'})
else                         : sys.exit('UNKNOWN DOMAIN')
del remove_days_with_nans
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
print('ALIGN DATASETS')
predictor, predictand = align_datasets(predictor, predictand, 'time')
del align_datasets
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
print('STANDARDIZE THE PREDICTORS')
predictor_stand = standardize(data_ref=predictor, data=predictor)
del standardize
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
print('SPLIT DATASETS (TRAIN/TEST)')
x_train, y_train, x_teste, y_keep  = split_data(train_exp, predictor, predictand)
del x_train, y_train, x_teste, predictor
gc.collect()

x_train, y_train, x_teste, y_teste = split_data(train_exp, predictor_stand, predictand)
del train_exp, predictor_stand, predictand, split_data, y_teste
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
if   (lossf == 'customBGamma') and (target_var == 'pr'):
  y_train = precipitation_NLL_trans(y_train, threshold_pr)

if (lossf == 'SqrMSE') and (target_var == 'pr'):
  y_train = y_train ** (1/2)

####################################################################################################################################################################################
####################################################################################################################################################################################
print('CONVERT TO NUMPY')
x_train_arr = xarray_to_numpy(x_train)
y_train_arr = xarray_to_numpy(y_train)
y_train_arr = y_train_arr.reshape(len(y_train_arr[:,0,0]),len(y_train_arr[0,:,0])*len(y_train_arr[0,0,:]))
del x_train, y_train
gc.collect()

x_teste_arr = xarray_to_numpy(x_teste)
del x_teste, xarray_to_numpy
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
print('SPLIT TRAIN DATASET (TRAIN/VALIDATION)')
XTrain, XValid, YTrain, YValid = train_test_split(x_train_arr, y_train_arr, train_size=0.9, test_size=0.1, random_state=42)
del x_train_arr, y_train_arr, train_test_split
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
print('DEFINE MODEL')
x_sizes = XTrain.shape;
y_sizes = YTrain.shape;

model_name = f'{ml_model}_{lossf}_{target_var}'
if   (target_var == 'tasmax') : [model, early_stopping, model_checkpoint, csv_logger] = def_model('CordexML_CNN_TASMAX', x_sizes, y_sizes, output_path, lossf, lrate, patience_es, model_name)
elif (target_var == 'pr')     : [model, early_stopping, model_checkpoint, csv_logger] = def_model('CordexML_CNN_PR',     x_sizes, y_sizes, output_path, lossf, lrate, patience_es, model_name)
del x_sizes, y_sizes, lrate, patience_es, def_model
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
print('TRAIN MODEL')
history = run_model(model, XTrain, YTrain, XValid, YValid, nbatch_size, nepochs, early_stopping, model_checkpoint, csv_logger)
del XTrain, YTrain, XValid, YValid, nbatch_size, nepochs, early_stopping, model_checkpoint, csv_logger, history, model
gc.collect()

####################################################################################################################################################################################
####################################################################################################################################################################################
print('PREDICT ON TEST DATASET')
x_teste_forecast = predict_model(output_path, ml_model, x_teste_arr, model_name, lossf, threshold_pr)

####################################################################################################################################################################################
###################################################################################################################################################################################
print('SAVE MODEL OUTPUT')
x_teste = y_keep.copy()

if (lossf == 'SqrMSE') and (target_var == 'pr'): x_teste_forecast = x_teste_forecast ** (2)

if   domain == 'ALPS_domain' : t = y_keep.sizes['time']; y = y_keep.sizes['y'];   x = y_keep.sizes['x']
else                         : t = y_keep.sizes['time']; y = y_keep.sizes['lat']; x = y_keep.sizes['lon']

if   target_var == 'tasmax' : x_teste.tasmax.values = x_teste_forecast.reshape(t,y,x)
elif target_var == 'pr'     : x_teste.pr.values     = x_teste_forecast.reshape(t,y,x)
del x_teste_forecast, target_var

filename = f'{output_path}/{model_name}_teste_for.nc'; x_teste.to_netcdf(filename); del filename, x_teste
filename = f'{output_path}/{model_name}_teste_obs.nc';  y_keep.to_netcdf(filename); del filename, y_keep
del output_path, ml_model

sys.exit()
