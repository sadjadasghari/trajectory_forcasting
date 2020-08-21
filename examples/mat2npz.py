import numpy as np
import scipy as sc
import os
import scipy.io as sio

data = np.load('../input/sim_data.npz')
Times = data['Times']  # 2000 samples x 200 time stamps
X = data['X']   # 2000x200x3 , 2000 trajectory samples of length 200 (time stamps) for xyz coordinates
N = len(Times)  # number of samples = 2000
size_perturb_trials = sio.loadmat('../input/Size_3DTrajectores_10Subs.mat')
print(size_perturb_trials.keys())
i,j = size_perturb_trials['Trajectories3D_SubByCondByTrials'].shape # number of subjects x number of experiments  (10x35)
subject = []
for ii in range(i):  # for each subject
    subject.append(size_perturb_trials['Trajectories3D_SubByCondByTrials'][ii]) # 1x35, each cell includes number of trials per blockx10
    for jj in range(j):
        trials = subject[ii][jj]  # each experiment, i.e. 1x45: number of attempts
        for t in trials[0]:
            trajectory = t[0][2:5]  # trial  # i.e. 173x10 which is Tx10 variables, we want variables [2:5] which is wrist XYZ coords
np.savez('Size_3DTrajectores_10Subs.npz',x=np.arange(3),y=np.ones((3,3)),z=np.array(3))