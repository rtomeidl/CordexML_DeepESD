####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
from netCDF4 import Dataset as nc

import xarray as xr
import numpy as np

import keras
import sys

####################################################################################################################################################################################
from .loss_predictions import *

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
def load_data(domain_path, domain, train_exp, pred_period, pred_type, pred_model):

  ##################################################################################################################################################################################
  if   pred_period == 'historical' :
    filename1 = f'{domain_path}/{domain}_domain/test/historical/predictors/{pred_type}/{pred_model}_1981-2000.nc'

  elif pred_period == 'mid_century' :
    filename1 = f'{domain_path}/{domain}_domain/test/mid_century/predictors/{pred_type}/{pred_model}_2041-2060.nc'

  elif pred_period == 'end_century' :
    filename1 = f'{domain_path}/{domain}_domain/test/end_century/predictors/{pred_type}/{pred_model}_2080-2099.nc'

  else :
    sys.exit('ERROR')

  ##################################################################################################################################################################################
  data     = xr.open_dataset(filename1, decode_times=False).load()

  ##################################################################################################################################################################################
  data = data.drop_vars('time_bnds', errors='ignore')

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

  return data

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
####################################################################################################################################################################################
####################################################################################################################################################################################
def predict_model(x_train_arr, output_path, model_name, lossf, input_path, train_exp, domain, target_var, threshold):

  ##################################################################################################################################################################################
  ### VERY LAZY CODE
  if   domain == 'ALPS' : xdomain='al'
  elif domain == 'NZ'   : xdomain='nz'
  elif domain == 'SA'   : xdomain='sa'
  else                  : sys.exit('ERROR')

  if   train_exp == 'Emulator_hist_future' : xtrain_exp='emu'
  elif train_exp == 'ESD_pseudo_reality'   : xtrain_exp='esd'
  else                                     : sys.exit('ERROR')

  if   target_var == 'pr'     : xtarget_var='pr_bgamma'
  elif target_var == 'tasmax' : xtarget_var='tasmax' 
  else                        : sys.exit('ERROR')

  ##################################################################################################################################################################################
  filename=f'{input_path}/{xdomain}_{xtrain_exp}_{xtarget_var}/results/BMDense_{lossf}_{target_var}'

  json_file = open(f'{filename}.json', 'r')
  loaded_model = json_file.read()
  json_file.close()

  if   lossf == 'customMSE'    : model = keras.models.model_from_json(loaded_model, custom_objects={'custom_fn': customMSE})
  elif lossf == 'SqrMSE'       : model = keras.models.model_from_json(loaded_model, custom_objects={'custom_fn': customMSE})
  elif lossf == 'customBGamma' : model = keras.models.model_from_json(loaded_model, custom_objects={'custom_fn': customBGamma})

  model.load_weights(f'{filename}.weights.h5')

  ##################################################################################################################################################################################
  forecast = np.squeeze(model.predict(x_train_arr, verbose=0))

  ##################################################################################################################################################################################
  if lossf == 'customBGamma' :
    # Get the parameters of the Bernoulli and gamma dists.
    dim_target = forecast.shape[1] // 3
    p     = forecast[:, :dim_target]
    shape = np.exp(forecast[:, dim_target:(dim_target*2)])
    scale = np.exp(forecast[:, (dim_target*2):])
    del forecast

    # Compute the ocurrence
    p_random = np.random.uniform(0, 1, p.shape)
    ocurrence = (p >= p_random) * 1

    # Compute the amount
    amount = np.random.gamma(shape=shape, scale=scale)
    del p, shape, scale, p_random

    # Correct the amount
    epsilon = 1e-06
    threshold = threshold - epsilon
    amount = amount + threshold

    # Combine ocurrence and amount
    forecast = ocurrence * amount
    del ocurrence, amount

  return forecast

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
def load_coords(domain):

  if   domain == 'ALPS' : 
    filename = '/media/vor_disk1/data/rtome/MachineLearning/aaa_CORDEX_ML_ADOMAINS/ALPS_domain/train/ESD_pseudo_reality/target/pr_tasmax_CNRM-CM5_1961-1980.nc'
    ds     = nc(filename)
    out_xx = ds['x'][:]
    out_yy = ds['y'][:]
    out_lo = ds['lon'][:]
    out_la = ds['lat'][:]
    ds.close()
  elif domain == 'NZ' :
    filename = '/media/vor_disk1/data/rtome/MachineLearning/aaa_CORDEX_ML_ADOMAINS/NZ_domain/train/ESD_pseudo_reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc'
    ds     = nc(filename)
    out_lo = ds['lon'][:]
    out_la = ds['lat'][:]
    out_xx = out_lo
    out_yy = out_la
    ds.close()
  elif domain == 'SA' :
    filename = '/media/vor_disk1/data/rtome/MachineLearning/aaa_CORDEX_ML_ADOMAINS/SA_domain/train/ESD_pseudo_reality/target/pr_tasmax_ACCESS-CM2_1961-1980.nc'
    ds     = nc(filename)
    out_lo = ds['lon'][:]
    out_la = ds['lat'][:]
    out_xx = out_lo
    out_yy = out_la
    ds.close()

  return out_lo, out_la, out_xx, out_yy

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
def save_results(model_name, domain, output_path, train_exp, target_var, pred_per, pred_typ, pred_mdl, xforecast_reshape, time, out_lo, out_la, out_xx, out_yy):

  if str(domain) == 'ALPS' : ndims = ['time','y',  'x']
  else                     : ndims = ['time','lat','lon']

  ncfile  =  nc(f'{output_path}/a{model_name}_Predictions_{domain}_{train_exp}_{target_var}_{pred_per}_{pred_typ}_{pred_mdl}.nc','w',format='NETCDF4')

  if str(domain) == 'ALPS' : 
    dm_bins = ncfile.createDimension('time',len(xforecast_reshape[:,0,0]))
  else :
    dm_bins = ncfile.createDimension('time', None)

  if domain == 'ALPS' :
    dm_lat  = ncfile.createDimension('y', len(xforecast_reshape[0,:,0]))
    dm_lon  = ncfile.createDimension('x', len(xforecast_reshape[0,0,:]))
  else :
    dm_lat  = ncfile.createDimension('lat', len(xforecast_reshape[0,:,0]))
    dm_lon  = ncfile.createDimension('lon', len(xforecast_reshape[0,0,:]))

  #################################################################################################################################################################################
  if   str(domain) == 'ALPS' :
    va_for = ncfile.createVariable(str(target_var),'f4',(str(ndims[0]),str(ndims[1]),str(ndims[2])), fill_value='NaN')
    if target_var == 'tasmax' :
      va_for.standard_name      = 'air_temperature'
      va_for.long_name          = 'Daily Maximum Near-Surface Air Temperature'
      va_for.units              = 'K'
      va_for.online_operation   = 'average'
      va_for.interval_operation = '1 d'
      va_for.interval_write     = '1 d'
      va_for.cell_methods       = 'time: maximum'
      va_for.grid_mapping       = 'Lambert_Conformal'
    va_for.coordinates        = 'lat lon'

    tt_for               = ncfile.createVariable(str(ndims[0]),'f8',(str(ndims[0])), fill_value='NaN')
    tt_for.axis          = 'T'
    tt_for.standard_name = 'time'
    tt_for.long_name     = 'Time axis'
    tt_for.time_origin   = '1949-12-01 00:00:00'
    tt_for.bounds        = 'time_bounds'
    tt_for.units         = 'days since 1949-12-01'
    tt_for.calendar      = 'gregorian'

    lo_for               = ncfile.createVariable('lon','f8',(str(ndims[1]),str(ndims[2])), fill_value='NaN')
    lo_for.standard_name = 'longitude'
    lo_for.long_name     = 'Longitude'
    lo_for.units         = 'degrees_east'
    lo_for.bounds        = 'bounds_lon'

    la_for               = ncfile.createVariable('lat','f8',(str(ndims[1]),str(ndims[2])), fill_value='NaN')
    la_for.standard_name = 'latitude'
    la_for.long_name     = 'Latitude'
    la_for.units         = 'degrees_north'
    la_for.bounds        = 'bounds_lat'

    xx_for               = ncfile.createVariable(str(ndims[2]),'f8',(str(ndims[2])), fill_value='NaN')
    xx_for.units         = 'km'
    xx_for.long_name     = 'x coordinate of projection'
    xx_for.standard_name = 'projection_x_coordinate'
    xx_for.axis          = 'X'

    yy_for               = ncfile.createVariable(str(ndims[1]),'f8',(str(ndims[1])), fill_value='NaN')
    yy_for.units         = 'km'
    yy_for.long_name     = 'y coordinate of projection'
    yy_for.standard_name = 'projection_y_coordinate'
    yy_for.axis          = 'Y'

  #################################################################################################################################################################################
  elif str(domain) == 'NZ' :
    if target_var == 'tasmax' :
      va_for = ncfile.createVariable(str(target_var),'f4',(str(ndims[0]),str(ndims[1]),str(ndims[2])), fill_value='NaN')
      va_for.standard_name      = 'air_temperature'
      va_for.long_name          = 'Daily Maximum Near-Surface Air Temperature'
      va_for.units              = 'K'
      va_for.cell_methods       = 'time: maximum over days'
    else :
      va_for = ncfile.createVariable(str(target_var),'f8',(str(ndims[0]),str(ndims[1]),str(ndims[2])), fill_value='NaN')

    tt_for               = ncfile.createVariable(str(ndims[0]),'i8',(str(ndims[0])))
    tt_for.units         = 'days since 1961-01-01'
    tt_for.calendar      = 'proleptic_gregorian'

    lo_for               = ncfile.createVariable(str(ndims[2]),'f4',str(ndims[2]), fill_value='NaN')
    lo_for.long_name     = 'longitude'
    lo_for.standard_name = 'longitude'
    lo_for.units         = 'degrees_east'
    lo_for.axis          = 'X'
    lo_for.bounds        = 'lon_bounds'

    la_for               = ncfile.createVariable(str(ndims[1]),'f4',str(ndims[1]), fill_value='NaN')
    la_for.long_name     = 'latitude'
    la_for.standard_name = 'latitude'
    la_for.units         = 'degrees_north'
    la_for.axis          = 'Y'
    la_for.bounds        = 'lat_bnds'

  #################################################################################################################################################################################
  elif str(domain) == 'SA' :
    if   target_var == 'pr' :
      va_for = ncfile.createVariable(str(target_var),'f8',(str(ndims[0]),str(ndims[1]),str(ndims[2])), fill_value='NaN')
      va_for.units = 'mm/day'
    elif target_var == 'tasmax' :
      va_for = ncfile.createVariable(str(target_var),'f8',(str(ndims[0]),str(ndims[1]),str(ndims[2])), fill_value=9.96921e+36)
      va_for.standard_name      = 'air_temperature'
      va_for.long_name          = 'Daily Maximum Near-Surface Air Temperature'
      va_for.units              = 'K'
      va_for.grid_mapping       = 'crs'
      va_for.cell_methods       = 'time: maximum (interval: 1 day time: mean'
      va_for.missing_value      = 9.96921e+36

    tt_for               = ncfile.createVariable(str(ndims[0]),'f4',(str(ndims[0])), fill_value='NaN')
    tt_for.standard_name = 'time'
    tt_for.bounds        = 'time_bnds'
    tt_for.axis          = 'T'
    tt_for.units         = 'minutes since 1961-01-01'
    tt_for.calendar      = 'standard'

    lo_for               = ncfile.createVariable(str(ndims[2]),'f4',str(ndims[2]), fill_value='NaN')
    lo_for.long_name     = 'longitude'
    lo_for.standard_name = 'longitude'
    lo_for.units         = 'degrees_east'
    lo_for.axis          = 'X'
    lo_for.bounds        = 'lon_bounds'

    la_for               = ncfile.createVariable(str(ndims[1]),'f4',str(ndims[1]), fill_value='NaN')
    la_for.long_name     = 'latitude'
    la_for.standard_name = 'latitude'
    la_for.units         = 'degrees_north'
    la_for.axis          = 'Y'
    la_for.bounds        = 'lat_bnds'

  #################################################################################################################################################################################
  va_for[:,:,:] = xforecast_reshape; del xforecast_reshape
  tt_for[:]     = time; del time

  if   domain == 'ALPS' :
    lo_for[:,:] = out_lo; del out_lo
    la_for[:,:] = out_la; del out_la
    xx_for[:]   = out_xx; del out_xx
    yy_for[:]   = out_yy; del out_yy
  elif domain == 'NZ' or domain == 'SA':
    lo_for[:] = out_lo; del out_lo
    la_for[:] = out_la; del out_la

  ncfile.close()

  return
