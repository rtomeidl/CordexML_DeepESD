########################################################################################################################################################################################################
########################################################################################################################################################################################################
########################################################################################################################################################################################################
from itertools import combinations

import sys
import os

########################################################################################################################################################################################################
########################################################################################################################################################################################################
########################################################################################################################################################################################################
ofile_n=1
for region in ['SA']:
  if region == 'SA' : models = ['ACCESS-CM2']

  for mtype in ['ESD_pseudo_reality','Emulator_hist_future']:

    for mtypef in ['imperfect']:

      for pvariable in ['pr','tasmax']:

        if str(pvariable) == 'pr' : lossf='customBGamma'
        else                      : lossf='customMSE'

        for mtime in ['historical','mid_century','end_century']:

          for omodel in models:
            print(region,mtype,pvariable,mtime,mtypef,omodel)

            with open('runfile_'+str(ofile_n).zfill(3),"w") as text_file:
              text_file.write('{\n')
              text_file.write('  "model_name"   : "DeepESD",\n')
              text_file.write('  "lossf"        : "'+str(lossf)+'",\n')
              text_file.write('  "target_var"   : "'+str(pvariable)+'",\n')
              text_file.write('  "threshold_pr" : 0.1,\n')
              text_file.write('  "input_path"   : "/media/vor_disk1/data/rtome/MachineLearning/aaa_CORDEX_ML_CNN_final",\n')
              text_file.write('  "output_path"  : "results",\n')
              text_file.write('  "domain_path"  : "/media/vor_disk1/data/rtome/MachineLearning/aaa_CORDEX_ML_ADOMAINS",\n')
              text_file.write('  "domain"       : "'+str(region)+'",\n')
              text_file.write('  "train_exp"    : "'+str(mtype)+'",\n')
              text_file.write('  "pred_period"  : "'+str(mtime)+'",\n')
              text_file.write('  "pred_type"    : "'+str(mtypef)+'",\n')
              text_file.write('  "pred_model"   : "'+str(omodel)+'"\n')
              text_file.write('}\n')
            ofile_n=ofile_n+1

print(ofile_n-1)
sys.exit()
