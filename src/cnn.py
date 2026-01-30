####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
import numpy as np

import keras
import sys
import gc

from .loss import *

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
def def_model(ml_model, x_sizes, y_sizes, output_path, lossf, lrate, patience_es, model_name):

  ##################################################################################################################################################################################
  ##################################################################################################################################################################################
  keras.backend.clear_session()

  ##################################################################################################################################################################################
  ##################################################################################################################################################################################
  ### CordexML_CNN_TASMAX
  if ml_model == 'CordexML_CNN_TASMAX':

    model   = keras.Sequential()
    input_  = keras.layers.Input(shape = (int(x_sizes[1]),int(x_sizes[2]),int(x_sizes[3])))
    conv1   = keras.layers.Conv2D( 50, kernel_size = (3,3), activation='relu', padding='valid', )(input_)
    conv1   = keras.layers.Conv2D( 25, kernel_size = (3,3), activation='relu', padding='valid', )(conv1)
    conv1   = keras.layers.Conv2D(  1, kernel_size = (3,3), activation='relu', padding='valid', )(conv1)
    flatten = keras.layers.Flatten()(conv1)
    output_ = keras.layers.Dense(units = int(y_sizes[1]), activation='linear', )(flatten)
    model   = keras.Model(inputs = [input_], outputs = [output_])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate = lrate), loss=customMSE)

  elif ml_model == 'CordexML_CNN_PR':

    model   = keras.Sequential()
    input_  = keras.layers.Input(shape = (int(x_sizes[1]),int(x_sizes[2]),int(x_sizes[3])))
    conv1   = keras.layers.Conv2D( 50, kernel_size = (3,3), activation='relu', padding='valid', )(input_)
    conv1   = keras.layers.Conv2D( 25, kernel_size = (3,3), activation='relu', padding='valid', )(conv1)
    conv1   = keras.layers.Conv2D( 10, kernel_size = (3,3), activation='relu', padding='valid', )(conv1)
    flatten = keras.layers.Flatten()(conv1)

    if   lossf == 'SqrMSE' :
      output_ = keras.layers.Dense(units = int(y_sizes[1]), activation='linear', )(flatten)
      model   = keras.Model(inputs = [input_], outputs = [output_])
      model.compile(optimizer=keras.optimizers.Adam(learning_rate = lrate), loss=customMSE)

    elif lossf == 'customBGamma' :
      dense1  = keras.layers.Dense(units = int(y_sizes[1]), activation='sigmoid', )(flatten)
      dense2  = keras.layers.Dense(units = int(y_sizes[1]), activation='linear',  )(flatten)
      dense3  = keras.layers.Dense(units = int(y_sizes[1]), activation='linear',  )(flatten)
      output_ = keras.layers.concatenate([dense1, dense2, dense3],axis=-1)
      model   = keras.Model(inputs = [input_], outputs = [output_])
      model.compile(optimizer=keras.optimizers.Adam(learning_rate = lrate), loss=customBGamma)

  else :
    sys.exit('UNKNOWN MODEL')

  ##################################################################################################################################################################################
  ### SAVE STRUCTURE AND PARAMETERS
  filename = f'{output_path}/{model_name}.json'
  model_json = model.to_json(indent=4)
  with open(filename, 'w') as json_file:
    json_file.write(model_json)

  ##################################################################################################################################################################################
  ### CALLBACKS
  estopping = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=patience_es, restore_best_weights=True, verbose=0, baseline=None, )

  ##################################################################################################################################################################################
  ### SAVE BEST MODEL
  filename = f'{output_path}/{model_name}.weights.h5'
  mcheckpoint = keras.callbacks.ModelCheckpoint(filepath=filename,monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, save_freq="epoch", )

  ##################################################################################################################################################################################
  ### KEEP TRAINING LOG
  filename = f'{output_path}/{model_name}_log.csv'
  clogger = keras.callbacks.CSVLogger(filename=filename, separator=',', append=False)

  ##################################################################################################################################################################################
  ### MODELS PLOT
  filename = f'{output_path}/{model_name}_network.png'
  keras.utils.plot_model(model, to_file=filename, show_shapes=True, expand_nested=True, show_layer_activations=True, show_trainable=True)

  ##################################################################################################################################################################################
  ### MODEL SUMMARY
  #model.summary()
  #sys.exit()

  return model, estopping, mcheckpoint, clogger

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
def run_model(model, x_train, y_train, x_valid, y_valid, nbatch_size, nepochs, early_stopping, model_checkpoint, csv_logger):

  history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=nbatch_size, epochs=nepochs, callbacks=[early_stopping, model_checkpoint, csv_logger], verbose=0)
  keras.backend.clear_session()

  return history

####################################################################################################################################################################################
####################################################################################################################################################################################
####################################################################################################################################################################################
def predict_model(output_path, ml_model, x_teste, model_name, lossf, threshold):
  json_file = open(f'{output_path}/{model_name}.json', 'r')
  loaded_model = json_file.read()
  json_file.close()

  if   lossf == 'customMSE'    : model = keras.models.model_from_json(loaded_model, custom_objects={'custom_fn': customMSE})
  elif lossf == 'SqrMSE'       : model = keras.models.model_from_json(loaded_model, custom_objects={'custom_fn': customMSE})
  elif lossf == 'customBGamma' : model = keras.models.model_from_json(loaded_model, custom_objects={'custom_fn': customBGamma})

  model.load_weights(f'{output_path}/{model_name}.weights.h5')

  x_teste_forecast = np.squeeze(model.predict(x_teste,verbose=0))

  if lossf == 'customBGamma' :
    # Get the parameters of the Bernoulli and gamma dists.
    dim_target = x_teste_forecast.shape[1] // 3
    p     = x_teste_forecast[:, :dim_target]
    shape = np.exp(x_teste_forecast[:, dim_target:(dim_target*2)])
    scale = np.exp(x_teste_forecast[:, (dim_target*2):])
    del x_teste_forecast
    gc.collect()

    # Compute the ocurrence
    p_random = np.random.uniform(0, 1, p.shape)
    ocurrence = (p >= p_random) * 1

    # Compute the amount
    amount = np.random.gamma(shape=shape, scale=scale)
    del p, shape, scale, p_random
    gc.collect()

    # Correct the amount
    epsilon = 1e-06
    threshold = threshold - epsilon
    amount = amount + threshold

    # Combine ocurrence and amount
    x_teste_forecast = ocurrence * amount
    del ocurrence, amount
    gc.collect()

  return x_teste_forecast

