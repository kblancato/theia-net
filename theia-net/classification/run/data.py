import sys
print(sys.version)
import os
import time
start_time = time.time()
import numpy as np

from sklearn.utils import shuffle

import modules.mutils as mutils
import modules.prepare_data as prepare_data

import torch
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mutils.device_status(device)


# PATHS ========================================================================
"""define paths and make new directory for run"""
RUN = int(sys.argv[1])
CODE_DIR = str(sys.argv[2])
HOME_DIR = str(sys.argv[3])

DATA_PATH = "/data"
SAVE_PATH = "/models"

SAMPLE_NAME = "/"+str(sys.argv[4])
SLABEL = str(sys.argv[5])
LABEL = "/"+str(sys.argv[5])+"/"
DTYPE = "/"+str(sys.argv[6])+"/"

paths = {}
paths['home_dir'] = HOME_DIR
paths['code_dir'] = CODE_DIR
paths['data_dir'] = HOME_DIR+DATA_PATH
paths['run_dir'] = HOME_DIR+SAVE_PATH+DTYPE+SAMPLE_NAME+LABEL

if not os.path.exists(paths['run_dir']+'run%s' % RUN):
    os.makedirs(paths['run_dir']+'run%s' % RUN)
paths['run_dir'] += "run%s/" % RUN


# LOGS =========================================================================
data_log = mutils.create_log(paths['run_dir'], "data_%s" % SLABEL)
timenow = time.time()-start_time
print('[INIT] run time: %.3f s' % timenow, file=data_log)


# LOAD AND PREPARE DATA ========================================================
data = np.load(paths['data_dir']+'/fluxes/%s_fluxes.npy' \
                % SAMPLE_NAME.replace("/", ""))
label = np.load(paths['data_dir']+'/labels/%s_%s.npy' % \
                (SAMPLE_NAME.replace("/", ""), LABEL.replace("/", "")))[1,:]
bad = np.load(paths['data_dir']+'/fluxes/%s_bad.npy' \
              % (SAMPLE_NAME.replace("/", "")))
stds = np.load(paths['data_dir']+'/fluxes/%s_stds.npy'\
               % (SAMPLE_NAME.replace("/", "")))

if len(bad) > 0:
    good = np.in1d(range(label.shape[0]), bad)
    data = data[~good, :]
    label = label[~good]
    stds = stds[~good]

# shuffle data
ind = np.arange(0, len(label), 1)
data, ind = shuffle(data, ind, random_state=0)
label = label[ind]
stds = stds[ind]


# train/val/test split
trainidx, validx, testidx = prepare_data.data_split(len(label), data_log)

X_train = prepare_data.scale_data(data[trainidx], 'train', paths['run_dir'])
X_val = prepare_data.scale_data(data[validx], 'val', paths['run_dir'])
X_test = prepare_data.scale_data(data[testidx], 'test', paths['run_dir'])

y_train = label[trainidx]
y_val = label[validx]
y_test = label[testidx]
n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)

# standard deviations
stds_train = prepare_data.scale_stds(stds[trainidx])
stds_val = prepare_data.scale_stds(stds[validx])
stds_test = prepare_data.scale_stds(stds[testidx])

# batch and prep for torch
BATCH_SIZE = 256

train_loader = prepare_data.data_for_torch(X_train, stds_train, y_train,
                                        BATCH_SIZE, device, data_log)
val_loader = prepare_data.data_for_torch(X_val, stds_val, y_val,
                                      BATCH_SIZE, device, data_log)
test_loader = prepare_data.data_for_torch(X_test, stds_test, y_test,
                                    X_test.shape[1], device, data_log)
val_pred_loader = prepare_data.data_for_torch(X_val, stds_val, y_val,
                                    X_val.shape[1], device, data_log)


# SAVE FOR TRAINING ============================================================
"save what is necessary for main training loops"
if not os.path.exists(paths['run_dir']+'tmp'):
    os.makedirs(paths['run_dir']+'tmp')

np.save(paths['run_dir']+'tmp/paths.npy', paths)
np.save(paths['run_dir']+'tmp/lens.npy', np.array([n_train, n_val, n_test]))
np.save(paths['run_dir']+'tmp/datashape.npy', data.shape)
np.save(paths['run_dir']+'tmp/y_test.npy', y_test)
np.save(paths['run_dir']+'tmp/y_val.npy', y_val)

torch.save(train_loader, paths['run_dir']+'tmp/'+'train_loader.pth')
torch.save(val_loader, paths['run_dir']+'tmp/'+'val_loader.pth')
torch.save(val_pred_loader, paths['run_dir']+'tmp/'+'val_pred_loader.pth')
torch.save(test_loader, paths['run_dir']+'tmp/'+'test_loader.pth')
