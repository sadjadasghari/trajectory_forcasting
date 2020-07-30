import numpy as np
import scipy as sc
import os
import scipy.io as sio

size_perturb_trials = scipy.io.loadmat('../input/Size_3DTrajectores_10Subs.mat')
print(size_perturb_trials.keys())