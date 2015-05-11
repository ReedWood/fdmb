"""@package docstring
Tshort for fdmb.arfit()

A two dimensionsional (dim=3) AR[3] (order=3) is fitted using nData=5000 data points.
"""

import pickle
import numpy as np
import fdmb


class MleRes:
    def __init__(self):
        self.mleA = 0
        self.mleQ = 0
        self.mleR = 0
        self.mleX = 0
        self.aThresh = 0
        self.pThresh = 0
        self.SNR = 0
        self.nData = 0


# Data parameters
dim = 2
order = 3
nData = 300
aThresh = 1e-7
pThresh = 1e-7
maxIter = 100000

trueA = np.array([[0.9000, 0,      -0.5000,  0.1000,  0,       0.1500],
                  [0.3500, 0.7000,  0.2000, -0.3000, -0.2500, -0.4000]])

# True parameters
# A1 =  0.9000         0
#       0.3500    0.7000

# A2 = -0.5000    0.1000
#       0.2000   -0.3000

# A3 =  0         0.1500
#      -0.2500   -0.4000

# Q =   1 0
#       0 1

# R =  25  0
#      0  25


# Load data
hiddenP5 = np.loadtxt('hidden-SNR-P5.dat')
obsP5 = np.loadtxt('obs-SNR-P5.dat')
obsNoiseP5 = np.loadtxt('obsNoise-SNR-P5.dat')

hiddenP125 = np.loadtxt('hidden-SNR-P125.dat')
obsP125 = np.loadtxt('obs-SNR-P125.dat')
obsNoiseP125 = np.loadtxt('obsNoise-SNR-P125.dat')


# Fit VAR model to data, least squares
lsA, lsQ, lsR = fdmb.arfit(hiddenP5, nData, dim, order)


# Fit VAR model to data, EM
print('Start EM on SNR=1/2')
mleP5A, mleP5Q, mleP5R, mleP5X, errA, errQ, errR \
      = fdmb.emfit(obsP5, nData, dim, order, aThresh, pThresh, maxIter, False, trueA)
#print('Start EM on SNR=1/8')
#mleP125A, mleP125Q, mleP125R, mleP125X, errA, errQ, errR \
      #= fdmb.emfit(obsP125, nData, dim, order, aThresh, pThresh, maxIter, False, trueA)

mleResList = []
mleResList.append(MleRes())
mleResList[-1].mleA = mleP5A
mleResList[-1].mleQ = mleP5Q
mleResList[-1].mleR = mleP5R
mleResList[-1].mleX = mleP5X
mleResList[-1].aThresh = aThresh
mleResList[-1].pThresh = pThresh
mleResList[-1].SNR = .5
mleResList[-1].nData = nData

#mleResList.append(MleRes())
#mleResList[-1].mleA = mleP125A
#mleResList[-1].mleQ = mleP125Q
#mleResList[-1].mleR = mleP125R
#mleResList[-1].mleX = mleP125X
#mleResList[-1].aThresh = aThresh
#mleResList[-1].pThresh = pThresh
#mleResList[-1].SNR = 1/8
#mleResList[-1].nData = nData


# Write results to disk
with open('fit_emfit_fixedData.pickle', 'wb') as f:
    pickle.dump(mleResList, f, pickle.HIGHEST_PROTOCOL)
