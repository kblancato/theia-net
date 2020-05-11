import os
import os.path as path
import re
import sys
import shutil
import glob
import numpy as np

import modules.evaluate as evaluate


# PATHS AND DATA ===============================================================
RUN = sys.argv[1]
DIR = sys.argv[2]
LABEL = sys.argv[3]
HYPER = sys.argv[4]

paths = np.load(DIR+'/tmp/paths.npy').item()

# combine logs
logs = ['time',
        'optim',
        'model',
        'performance']

if path.exists(paths['run_dir']+"performance_%s.log" % LABEL):
    pass
else:
    for log in logs:
        files = glob.glob(paths['run_dir']+'%s*.log' %log)
        nums = []
        for f in files:
            try:
                nums.append(int(re.findall(r'\d+', f.split('/')[-1])[0]))
            except:
                pass
        idx = np.argsort(nums)
        files = np.array(files)[idx]
    
        with open(paths['run_dir']+"%s_%s.log" %(log, LABEL), "wb") as outfile:
            for f in files:
                with open(f, "rb") as infile:
                    outfile.write(infile.read())

performance_log = paths['run_dir']+'performance_%s.log' % LABEL
with open(performance_log) as f:
    metrics = f.readlines()

pid = np.array([int(re.search('PID: (.*), r2:', p).group(1))
       for p in metrics])
r2 = np.array([float(re.search('r2: (.*), bias', p).group(1))
                                    for p in metrics])
bias = np.array([np.abs(float(re.search('bias: (.*), rms', p).group(1)))
                                    for p in metrics])
rms = np.array([float(re.search('rms: (.*)', p).group(1))
                                    for p in metrics])


# CHOOSE MODEL =================================================================
y_val_true = np.load(paths['run_dir']+'tmp/y_val.npy')
y_val_true = evaluate.inverse_scale(paths['run_dir'], 'val', y_val_true)
y_mean = np.mean(y_val_true)
y_std = np.std(y_val_true)

bias_cut, rms_cut = .1, .5
bias_idx = np.where(np.array(bias) <= bias_cut*y_mean)[0]
rms_idx = np.where(np.array(rms) <= rms_cut*y_std)[0]
good = np.intersect1d(bias_idx, rms_idx)

if len(good) > 0:
    model_idx = np.argsort(r2[good])[::-1]
    sorted_models = pid[good][model_idx]
    sorted_r2 = r2[good][model_idx]
    best_model = pid[good][model_idx[0]]

# if no model meets criteria, just rank by r2
if len(good) == 0:
    model_idx = np.argsort(r2)[::-1]
    sorted_models = pid[model_idx]
    sorted_r2 = r2[model_idx]
    best_model = pid[model_idx[0]]


# PLOT AND SAVE RESULTS ========================================================
ranked = np.stack((sorted_models, sorted_r2))
np.save(paths['run_dir']+'ranked_models.npy', ranked)

# examine top 10 ranked models
top_dir = paths['run_dir']+'top_models/'
try:
    shutil.rmtree(top_dir)
except OSError:
    pass
os.mkdir(top_dir)

for n,m in enumerate(ranked[0,0:10]):
    plots = glob.glob(paths['run_dir']+'pid%s/*.png' % int(m))
    for p in plots:
        shutil.copy(p, top_dir)
        os.rename(top_dir+p.split('/')[-1],
                  top_dir+p.split('/')[-1].split('.')[0]+'_%s.png' % int(n))

param_file = 'hyperparams%s.npy' %HYPER
param_dict = np.load(paths['code_dir']+'/hyperparams/'+param_file)[pid]

metrics = {'pid':pid,
           'r2':r2,
           'bias':bias,
           'rms':rms}

evaluate.plot_hyperparams(metrics, param_dict, LABEL, paths['run_dir'])

best_dir =  path.abspath(path.join(__file__ ,
                    paths['run_dir']+"../.."))+'/best/%s' %LABEL

try:
    shutil.rmtree(best_dir)
except:
    pass
try:
    shutil.rmtree(paths['run_dir']+'%s' % LABEL)
except:
    pass

shutil.copytree(paths['run_dir']+'pid%s' %best_model,
                paths['run_dir']+'%s' % LABEL)
shutil.copytree(paths['run_dir']+'%s' % LABEL, best_dir)