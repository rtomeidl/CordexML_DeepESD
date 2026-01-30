####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
import xarray as xr
import numpy as np

import sys

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
def load_data(input_path, train_exp, domain, input_dir, target_var="None"):

  if   domain == 'ALPS_domain' : smodel = 'CNRM-CM5_1961-1980'
  elif domain == 'NZ_domain'   : smodel = 'ACCESS-CM2_1961-1980'
  elif domain == 'SA_domain'   : smodel = 'ACCESS-CM2_1961-1980'
  else                         : sys.exit('UNKNOWN DOMAIN')

  if train_exp == 'Emulator_hist_future' : smodel = smodel + '_2080-2099'

  if input_dir == 'target' : smodel = 'pr_tasmax_' + smodel

  filename = f'{input_path}/{domain}/train/{train_exp}/{input_dir}/{smodel}.nc'
  data     = xr.open_dataset(filename).load()

  if input_dir == 'target' :
    if   target_var == 'tasmax' : data=data.drop_vars(["pr"])
    elif target_var == 'pr'     : data=data.drop_vars(["tasmax"])
    else                        : sys.exit('UNKNOWN TARGET')

  if domain == 'SA_domain' :
    data=data.drop_vars(["time_bnds"])
    if input_dir == 'target' :
      data=data.drop_vars(["lon_bnds"])
      data=data.drop_vars(["lat_bnds"])
      data=data.drop_vars(["crs"])

  return data

####################################################################################################################################################################################
### FROM DEEP4DOWNSCALING ##########################################################################################################################################################
####################################################################################################################################################################################
def remove_days_with_nans(data, coord_names):
# Get Time Indices With Zero Null Values
  nans_indices = data.isnull()
  nans_indices = nans_indices.sum(dim=(coord_names['lat'],coord_names['lon'])).to_array().values
  nans_indices = np.logical_or.reduce(nans_indices, axis=0)
  nans_indices = ~nans_indices

# Filter The Dataset
  data = data.sel(time=nans_indices)

# Log The Operation
  # # # if np.sum(nans_indices) == len(nans_indices): print( ' ... There are no observations containing null values')
  # # # else                                        : print(f' ... Removing {np.sum(nans_indices)} observations contaning null values')

  return data

####################################################################################################################################################################################
### FROM DEEP4DOWNSCALING ##########################################################################################################################################################
####################################################################################################################################################################################
def align_datasets(data_1, data_2, coord):
  data_1 = data_1.sel(time=np.isin(data_1[coord].values,data_2[coord].values))
  data_2 = data_2.sel(time=np.isin(data_2[coord].values,data_1[coord].values))

  return data_1, data_2

####################################################################################################################################################################################
### FROM DEEP4DOWNSCALING ##########################################################################################################################################################
####################################################################################################################################################################################
def standardize(data_ref, data):
  mean = data_ref.mean('time')
  std  = data_ref.std('time')

  data_stand = (data - mean) / std

  return data_stand

####################################################################################################################################################################################
### FROM DEEP4DOWNSCALING ##########################################################################################################################################################
####################################################################################################################################################################################
def xarray_to_numpy(data, ignore_vars: list[str]=None):
  final_data = []
  data_vars = [i for i in data.data_vars]
  if ignore_vars:
    data_vars = [i for i in data.data_vars if i not in ignore_vars]

  for var_convert in data_vars:
    final_data.append(data[var_convert].to_numpy())

  if len(data_vars) == 1:
    final_data = final_data[0]
  else:
    final_data = np.stack(final_data, axis=3)

  return final_data

####################################################################################################################################################################################
### FROM DEEP4DOWNSCALING ##########################################################################################################################################################
####################################################################################################################################################################################
def precipitation_NLL_trans(data, threshold):
  data_final = data.copy(deep=True)

  epsilon = 1e-06
  threshold = threshold - epsilon
  data_final = data_final - threshold
  data_final = xr.where(cond=data_final<0, x=0, y=data_final)

  return data_final

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
def split_data(train_exp, predictor, predictand):
  if   train_exp == 'ESD_pseudo_reality'   :
    years_train = list(range(1961, 1980+1))
    years_teste = list(range(1961, 1980+1))

  elif train_exp == 'Emulator_hist_future' :
    years_train = list(range(1961, 1980+1)) + list(range(2080, 2099+1))
    years_teste = list(range(1961, 1980+1)) + list(range(2080, 2099+1))

  elif train_exp == 'EXP'                  :
    years_train = list(range(1961, 1962))
    years_teste = list(range(1962, 1963))

  else                                     :
    sys.exit('UNKNOWN SPLIT')

  x_train =  predictor.sel(time=np.isin( predictor['time'].dt.year, years_train))
  y_train = predictand.sel(time=np.isin(predictand['time'].dt.year, years_train))

  x_teste =  predictor.sel(time=np.isin( predictor['time'].dt.year, years_teste))
  y_teste = predictand.sel(time=np.isin(predictand['time'].dt.year, years_teste))

  return x_train, y_train, x_teste, y_teste
