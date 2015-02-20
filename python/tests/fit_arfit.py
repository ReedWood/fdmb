"""@package docstring
Test for fdmb.arfit()

A two dimensionsional (dim=3) AR[3] (order=3) is fitted using nData=5000 data points.
"""

import ctypes as ct
import numpy as np
import sys
import fdmb


# Generate dim 2 AR[3] data
dim = 2
order = 3
nData = 5000

A1 = np.array([[.9, 0], [.4, .8]])
A2 = np.array([[-.6, 0], [0, -.4]])
A3 = np.array([[0, .3], [0, 0]])

print("True VAR parameters")
print(A1, end='\n')
print(A2, end='\n')
print(A3, end='\n\n')
sys.stdout.flush()

data = np.zeros([nData, dim])
data[:3, :] = np.random.randn(3, 2)
for t in np.arange(order, nData, 1):
    data[t, :] = A1.dot(data[t-1, :]) + A2.dot(data[t-2, :]) + \
                 A3.dot(data[t-3, :]) + np.random.randn(dim)


# Fit VAR model to data
arCoeff, noiseCov, arCov = fdmb.arfit(data, nData, dim, order)

print(np.around(arCoeff, decimals=1))
print(np.around(noiseCov, decimals=1))
print(np.around(arCov, decimals=3))
