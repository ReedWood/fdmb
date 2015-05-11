"""@package docstring
Tshort for fdmb.arfit()

A two dimensionsional (dim=3) AR[3] (order=3) is fitted using nData=5000 data points.
"""

import numpy as np
import fdmb


# Data parameters
dim = 2
order = 3
nDataShort = 3000
nDataLong  30000
aThresh = 1e-5
pThresh = 1e-7
maxIter = 100000

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
hiddenShort = np.loadtxt('obsShort.dat')
hiddenLong = np.loadtxt('obsLong.dat')
obsShort = np.loadtxt('obsShort.dat')
obsLong = np.loadtxt('obsLong.dat')


# Fit VAR model to data, least squares
shortLSA, shortLSQ, shortLSR = fdmb.arfit(hiddenShort, nDataShort, dim, order)
longLSA, longLSQ, longLSR = fdmb.arfit(hiddenLong, nDataLong, dim, order)

# Fit VAR model to data, EM
shortMLEAc, shortMLEQc, shortMLERc, shortX, errA, errQ, errR \
      = fdmb.emfit(obsShort, nDataShort, dim, order, aThresh, pThresh, maxIter, False)
longMLEAc, longMLEQc, longMLERc, longX, errA, errQ, errR \
      = fdmb.emfit(obsLong, nDataLong, dim, order, aThresh, pThresh, maxIter, False)


## Print out test results
#print('Estimated VAR parameter values')
#print(arCoeff[0][0], end='\n\n')
#print(arCoeff[0][1], end='\n\n')
#print(arCoeff[0][2], end='\n\n')

#print(arCoeff[1], end='\n\n')
#print(arCoeff[2], end='\n\n')