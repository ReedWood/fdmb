"""@package docstring
Test for fdmb.arfit()

A two dimensionsional (dim=3) AR[3] (order=3) is fitted using nData=5000 data points.
"""

import numpy as np
import fdmb


# Generate dim 2 AR[3] data
dim = 2
order = 3
nData = 10000
aThresh = .001
pThresh = .001
maxIter = 10000

A1 = np.array([[.9, 0], [.4, .8]])
A2 = np.array([[-.6, 0], [0, -.4]])
A3 = np.array([[0, .3], [0, 0]])

data = np.zeros([nData, dim])
data[:3, :] = np.random.randn(3, 2)
for t in np.arange(order, nData, 1):
    data[t, :] = A1.dot(data[t-1, :]) + A2.dot(data[t-2, :]) + \
                 A3.dot(data[t-3, :]) + np.random.randn(dim)

# Generate observations
obs = data + .2*np.random.randn(*data.shape)

# Fit VAR model to data
arCoeff = fdmb.emfit(obs, nData, dim, order, aThresh, pThresh, maxIter)

testsum = np.sum(np.round(A1-arCoeff[0], decimals=1)) + \
          np.sum(np.round(A2-arCoeff[1], decimals=1)) + \
          np.sum(np.round(A3-arCoeff[2], decimals=1))

if testsum > 0:
    print("Test failed")
else:
    print("Test passed")
