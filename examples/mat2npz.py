import numpy as np
import scipy as sc
import os
import scipy.io as sio
import matplotlib.pyplot as plt

data = np.load('../input/sim_data.npz')
Times = data['Times']  # 2000 samples x 200 time stamps
X = data['X']   # 2000x200x3 , 2000 trajectory samples of length 200 (time stamps) for xyz coordinates
N = len(Times)  # number of samples = 2000
size_perturb_trials = sio.loadmat('../input/Size_3DTrajectores_10Subs.mat')
print(size_perturb_trials.keys())
i,j = size_perturb_trials['Trajectories3D_SubByCondByTrials'].shape # number of subjects x number of experiments  (10x35)
subject = []
trajectories = []
min_shape = 225
max_shape = 231
for ii in range(i):  # for each subject
    subject.append(size_perturb_trials['Trajectories3D_SubByCondByTrials'][ii]) # 1x35, each cell includes number of trials per blockx10
    for jj in range(j):
        trials = subject[ii][jj]  # each experiment, i.e. 1x45: number of attempts
        for t in trials[0]:
            trajectory = t[:, 0:4]  # trial  # i.e. 173x10 which is Tx10 variables, we want variables [2:5] which is wrist XYZ coords
            start_ind = round((trajectory.shape[0] - 150) / 2)   # finding start ind
            end_ind = start_ind + 150  # getting a subset of trajectory with length of 150, from the middle of traj.
            trajectories.append(trajectory[start_ind:end_ind, :])
            # if min_shape > t.shape[0]:
            #     min_shape = t.shape[0]
            # if max_shape < t.shape[0]:
            #     max_shape = t.shape[0]

# np.savez('Size_3DTrajectores_10Subs.npz', X=trajectories, Times=Times)  # x=np.arange(3),y=np.ones((3,3)),z=np.array(3))
print(max_shape, min_shape)
plt.hist([t.shape[0] for t in trajectories], bins=50, density=1)       # matplotlib version (plot)

# Compute the histogram with numpy and then plot it

(n, bins) = np.histogram([t.shape[0] for t in trajectories], bins=50, density=True)  # NumPy version (no plot)

plt.plot(.5*(bins[1:]+bins[:-1]), n)

plt.show()