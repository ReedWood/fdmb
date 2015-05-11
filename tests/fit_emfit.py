"""@package docstring
Test for fdmb.arfit()

A two dimensionsional (dim=3) AR[3] (order=3) is fitted using nData=5000 data points.
"""

import numpy as np
import fdmb


# Generate dim 2 AR[3] data
dim = 2
order = 3
nData = 30000
aThresh = .00001
pThresh = .00001
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
obsNoise = 3*np.random.randn(*data.shape)
obs = data + obsNoise

# Fit VAR model to data
arCoeff = fdmb.emfit(obs, nData, dim, order, aThresh, pThresh, maxIter)

# Print out test results
print('Estimated VAR parameter values')
print(arCoeff[0][0], end='\n\n')
print(arCoeff[0][1], end='\n\n')
print(arCoeff[0][2], end='\n\n')

testsum = np.sum(np.round(A1-arCoeff[0][0], decimals=1)) + \
          np.sum(np.round(A2-arCoeff[0][1], decimals=1)) + \
          np.sum(np.round(A3-arCoeff[0][2], decimals=1))

if testsum > 0:
    print("Test failed")
else:
    print("Test passed")
