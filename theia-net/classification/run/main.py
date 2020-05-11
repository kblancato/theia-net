import sys
print(sys.version)
import os
import time
start_time = time.time()
import numpy as np

import modules.mutils as mutils
import modules.model as model
import modules.train as train
import modules.evaluate as evaluate

import torch
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mutils.device_status(device)


# READ IN DATA AND PATHS =======================================================
RUN = sys.argv[1]
DIR = sys.argv[2]
LABEL = sys.argv[3]
HYPER = sys.argv[4]
pid = int(sys.argv[5])

paths = np.load(DIR+'/tmp/paths.npy').item()

train_loader = torch.load(paths['run_dir']+'tmp/'+'train_loader.pth')
val_loader = torch.load(paths['run_dir']+'tmp/'+'val_loader.pth')

lens = np.load(paths['run_dir']+'tmp/lens.npy')
n_train, n_val = lens[0], lens[1]
ts_len = np.load(paths['run_dir']+'tmp/datashape.npy')[1]

time_log = mutils.create_log(paths['run_dir'], "time%s_%s" %(pid, LABEL))
optim_log = mutils.create_log(paths['run_dir'], "optim%s_%s" %(pid, LABEL))
model_log = mutils.create_log(paths['run_dir'], "model%s_%s" %(pid, LABEL))
performance_log = mutils.create_log(paths['run_dir'],
                                    "performance%s_%s" %(pid, LABEL))


# HYPERPARAMETERS ==============================================================
paths['save_dir'] = paths['run_dir'] + 'pid%s/' % pid

if not os.path.exists(paths['save_dir']):
    os.makedirs(paths['save_dir'])

param_file = 'hyperparams%s.npy' %HYPER
param_dict = np.load(paths['code_dir']+'/hyperparams/'+param_file)[pid]
print('[HYPERPARAM %s] params: %s' % (pid, param_dict), file=optim_log)


# LOAD MODEL ===================================================================
net = models.CNN(num_in=ts_len,
                 n_classes=3,
                 log=model_log,
                 kernel1=int(param_dict['KERNEL_1']),
                 kernel2=int(param_dict['KERNEL_2']),
                 padding1=int(param_dict['PADDING_1']),
                 padding2=int(param_dict['PADDING_2']),
                 stride1=int(param_dict['STRIDE_1']),
                 stride2=int(param_dict['STRIDE_2']),
                 dropout=float(param_dict['DROPOUT']) )

net.to(device)

print(net)
print(net, file=model_log)


# TRAIN ========================================================================
train_log = mutils.create_log(paths['save_dir'], "train_%s" % LABEL)

N_EPOCHS = 800
N_STOP = 50
TOL = .01

optimizer = torch.optim.AdamW(net.parameters(),
                             lr=float(param_dict['LR']),
                             weight_decay=float(param_dict['WD']),
                             eps=float(param_dict['EPS']))

loss_func = torch.nn.CrossEntropyLoss()

train.training(model=net,
               device=device,
               n_stop=N_STOP,
               tol=TOL,
               optimizer=optimizer,
               loss_func=loss_func,
               n_epochs=N_EPOCHS,
               train_loader=train_loader,
               n_train=n_train,
               val_loader=val_loader,
               n_val=n_val,
               save_path=paths['save_dir'],
               log=train_log, pid=pid)


# EVALUATE =====================================================================
model = net
model.load_state_dict(torch.load(paths['save_dir']+'model.pt'))

# make predictions
val_pred_loader = torch.load(paths['run_dir']+'tmp/'+'val_pred_loader.pth')
y_val_pred = evaluate.predictions(device, model, val_pred_loader)
y_val_probs = evaluate.probs(device, model, val_pred_loader)
y_val = np.load(paths['run_dir']+'tmp/y_val.npy')

test_loader = torch.load(paths['run_dir']+'tmp/'+'test_loader.pth')
y_test_pred = evaluate.predictions(device, model, test_loader)
y_test_probs = evaluate.probs(device, model, test_loader)
y_test = np.load(paths['run_dir']+'tmp/y_test.npy')

# save for plotting later
np.save(paths['save_dir']+'y_val_pred.npy', y_val_pred)
np.save(paths['save_dir']+'y_val_true.npy', y_val)
np.save(paths['save_dir']+'y_val_probs.npy', y_val_probs)
np.save(paths['save_dir']+'y_test_pred.npy', y_test_pred)
np.save(paths['save_dir']+'y_test_true.npy', y_test)
np.save(paths['save_dir']+'y_test_probs.npy', y_test_probs)

# compute evaluation metrics
auroc = evaluate.AUROC(y_val, y_val_probs)
average_precision = evaluate.average_precision(y_val, y_val_probs)
accuracy = evaluate.accuracy(y_val_pred, y_val)

metrics = {'auroc':auroc,
           'average_precision':average_precision,
           'accuracy':accuracy}

# make and save plots
evaluate.plot_classification(paths['save_dir'], pid,
                             y_val_pred, y_val, y_val_probs)

# SAVE =========================================================================
"save model performance"
print('PID: %s, auroc: %.3f, average_precision: %.3f, accuracy: %.3f' %(pid,auroc,
                                average_precision,accuracy), file=performance_log)

timenow = time.time()-start_time
print('[FINISH %s] run time: %.3f s' % (pid, timenow), file=time_log)
