# theia-net
- A data-driven deep learning method to determine stellar properties from light curve data alone, named after the Greek goddess Theia, the titaness of sight and heavenly light, associated with prophecy, wisdom, and the shimmering sky. 

- The 1D CNN code includes a classification model to predict stellar evolutionary state, as well as a regression model to predict stellar properties (e.g. rotation period, surface gravity, and temperature) from light curves of various baselines and cadences, optimizing model performance over a grid of relevant hyperparameters. 

- The models presented here, and in the accompanying paper, are trained using *Kepler* long-cadence light curves and various stellar property catalogs. However, the modeling approach can be generalized to other surveys and stellar datasets. 

- See the paper, "*Data-driven derivation of stellar properties from photometric time series data using convolutional neural networks*" by **Kirsten Blancato** (Columbia), **Melissa Ness** (Columbia/Flatiron), and **Dan Huber** (IfA), at [arXiv:XXXX](https://arxiv.org/list/astro-ph/recent).

## Code authors
Kirsten Blancato, knb2128-at-columbia-dot-edu

## Usage
This code is not meant to be used as a package, but as a template for further use and development. It includes a minimum working example of two use cases in the associated paper: the classification of stellar evolutionary state and the prediction of stellar rotation period from 27-day *Kepler* light curves. If you are interested in using or adapting parts of the code, please feel free to reach out to the authors. 

### Dependencies
- [PyTorch](https://pytorch.org/) </br>
- [scikit-learn](https://scikit-learn.org/stable/) </br>
- NumPy </br>
- SciPy </br>
- Matplotlib </br>
- Assumes access to GPUs for model training </br>
- Uses [disBatch](https://github.com/flatironinstitute/disBatch), a code for the distributed processing of a batch of tasks, to train a grid of models over the set of hyperparameters

### Code structure
For both the classification and regression models, the main model and training code in located in */modules/* directory. The code to prepare the data, train the models and select the best performing model is located in the */run/* directory. Example submit scripts for each problem are provides in the */example/* directory. 

### Data
We put example data for the classification of evolutionary state, as well as the prediction of stellar rotation period from 27-day *Kepler* data, online for download at [http://user.astro.columbia.edu/~kblancato/data/theia-net](http://user.astro.columbia.edu/~kblancato/data/theia-net).

### Steps
1. *make_run.sh*: Generate the submission scripts, specifying the necessary paths, datasets, labels, and hyperparameter files. </br>
2. *data_mcquillan_prot_27.sh (data.py)*: Prepare the data for training. </br>
3. *disbatch_mcquillan_prot_27 (main.py)*: Train a grid of models over the set of specified hyperparameters. </br>
4. *select_mcquillan_prot_27.sh (select.py)*: Evaluate the model performance. 
