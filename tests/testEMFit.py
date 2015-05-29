import numpy as np
import fdmb
import itertools


# Header
print('fdmb test for the EM fit implementation')
print('Documentation on the data can be found in ar2Data.README', end='\n\n')

# Model parameters
obs = np.loadtxt('ar2DataObs.dat')
nData = 5000
dim = 2
order = 2
aThresh = 1e-8
pThresh = 1e-10
maxIter = np.int32(1e6)
estError = False


# Expected result
targetA = np.array([[[1.30550248,  0.28797118],
                     [-0.02384243, 1.71119078]],
                    [[-8.02262977e-01,  5.75953420e-03],
                     [-4.26190044e-04, -7.90536237e-01]]])
targetQ = np.array([[1.09429302,  -0.00408784],
                    [-0.00408784, 0.88341843]])
targetR = np.array([[10.8341601,  -0.11942269],
                    [-0.11942269, 11.0743018]])

# Fit and results
print('Start fit')
arCoeff = fdmb.emfit(obs, nData, dim, order, aThresh, pThresh, maxIter, estError)

estA = np.asanyarray(arCoeff[0])
estQ = arCoeff[1]
estR = arCoeff[2]


# Compare expectation to actual result
passA = np.all((np.around(targetA, 8)-np.around(estA, 8)) < np.finfo(float).eps)
passQ = np.all((np.around(targetQ, 8)-np.around(estQ, 8)) < np.finfo(float).eps)
passR = np.all((np.around(targetR, 8)-np.around(estR, 8)) < np.finfo(float).eps)

labels = ['A: Transition matrix',
          'Q: Driving noise covariance',
          'R: Observational noise covariance']
failIdx = [not passA, not passQ, not passR]

if not np.any(failIdx):
    print('EM fit installation OK')

else:
    print('EM fit test failed for')
    failMsg = list(itertools.compress(data=labels, selectors=failIdx))
    for msg in failMsg:
        print('\t%s' % msg)

print('\nResult of the fit')
print(labels[0])
print(estA, end='\n\n')

print(labels[1])
print(estQ, end='\n\n')

print(labels[2])
print(estR)
