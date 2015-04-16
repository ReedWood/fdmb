"""@package docstring
Python wrapper for libfit.

This package defines function and return types for functions in libfit.
"""

import ctypes as ct
import numpy as np


# Load libfdmb and define input types
ct.cdll.LoadLibrary("libfdmbpy.so")
libfdmb = ct.CDLL("libfdmbpy.so")


# py_arfit: argtypes, restypes
libfdmb.py_arfit.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                             ct.c_int32,
                             ct.c_int32,
                             ct.c_int32,
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                             ]

libfdmb.py_arfit.restypes = np.int32


# Interface for libfdmb.py_arfit
def arfit(data, nData, dim, order):
    """
    Fit parameters of a vector autoregressive (VAR) process.

    Input:
    data: Measurements
    nData: Number of measurement times
    dim: Number of measured channels, dimension of the VAR
    order: Order of the VAR

    Return:
    arCoeff: List of lengths <order> of VAR parameters
    noiseCoeff: Covariance matrix of the driving noise eta
    covariance: The vector autocovariance of the process

    The parameters A_1, ..., A_order of the <dim> dimensional
    VAR[order] model
       x(t) = A_1.dot(x(t-1)) + A_2.dot(x(t-2)) +
              ... + A_order.dot(x(t-n)) + eta(t)
    are fitted to <data>. x(t) is a <dim>-dimensional vector. There
    are <nData> man x(t)-vectors and <order> many parameter matrices
    A_x to A_order. eta is the so called driving noise and is assumed
    to be normally distributed with mean zero and covariance S.

    <noiseCoeff> is an estimate of S. <covariance> is the covariance
    of the vector-valued process.
    """

    # Ensure correct dtype and layout of data
    m_data = np.array(data, dtype=np.float64, order='C')
    m_arCoefficients = np.zeros((dim, dim*order), dtype=np.float64, order='C')
    m_noiseCovariance = np.zeros((dim, dim), dtype=np.float64, order='C')
    m_processCovariance = np.zeros((dim*order, dim*order), dtype=np.float64, order='C')

    status = libfdmb.py_arfit(m_data, nData, dim, order,
                              m_arCoefficients, m_noiseCovariance, m_processCovariance)

    # Split coefficient matrix
    m_arCoeffArray = np.hsplit(m_arCoefficients, order)

    return m_arCoeffArray, m_noiseCovariance, m_processCovariance


# py_emfit: argtypes, restypes
libfdmb.py_emfit.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),  # data
                             ct.c_int32,                                        # nData
                             ct.c_int32,                                        # dim
                             ct.c_int32,                                        # order
                             ct.c_double,                                       # aThresh
                             ct.c_double,                                       # pThresh
                             ct.c_int32,                                        # maxIter
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),  # m_arCoefficients
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),  # m_dynNoiseCov
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),  # m_obsNoiseCov
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),  # m_hiddenStates
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=1),  # m_estimationError
                             ct.c_bool,                                         # estError
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),  # initTransitionMatrix
                             ]

libfdmb.py_emfit.restypes = np.int32


def emfit(data, nData, dim, order, aThresh, pThresh=1e-6, maxIter=np.int(1e-4),
          estError=False, initTransitionMatrix=np.nan*np.eye(1)):
    # Ensure correct dtype and layout of data
    m_data = np.array(data, dtype=np.float64, order='C')
    m_arCoefficients = np.zeros((dim*order, dim*order), dtype=np.float64, order='C')
    m_dynNoiseCov = np.zeros((dim*order, dim*order), dtype=np.float64, order='C')
    m_obsNoiseCov = np.zeros((dim, dim), dtype=np.float64, order='C')
    m_hiddenStates = np.zeros((dim*order, nData+1), dtype=np.float64, order='C')
    m_estimationError = np.zeros(dim**2*(order+2), dtype=np.float64, order='C')

    status = libfdmb.py_emfit(m_data, nData, dim, order,
                              aThresh, pThresh, maxIter,
                              m_arCoefficients, m_dynNoiseCov, m_obsNoiseCov,
                              m_hiddenStates, m_estimationError,
                              estError, initTransitionMatrix)

    # Split coefficient matrix
    m_arCoeffArray = np.hsplit(m_arCoefficients[:dim, :], order)

    # Grab relevant part of the dynamic noise covariance matrix q
    m_dynNoiseCov = m_dynNoiseCov[:dim, :dim]

    # Split error/uncertainty of the parameter estimates
    # What we calculate here is the variance of the distribution of
    #   ThetaEst - Theta
    # where ThetaEst is the estimate of a parameter and Theta its true value[1].
    #
    # [1] Shumway and Stoffer, Time series analysis and its application, page 344
    m_estimationError = m_estimationError/np.sqrt(nData)

    m_arCoeffError = []
    m_dynNoiseError = np.zeros([dim, dim])
    m_obsNoiseError = np.zeros([dim, dim])
    for i in range(0, order):
        m_arCoeffError.append(np.reshape(m_estimationError[i*dim**2:(i+1)*dim**2], (dim, dim)))
    m_dynNoiseError = np.reshape(m_estimationError[dim**2*order:dim**2*order+dim**2], (dim, dim))
    m_obsNoiseError = np.reshape(m_estimationError[dim**2*order+dim**2:dim**2*order+2*dim**2], (dim, dim))

    return m_arCoeffArray, m_dynNoiseCov, m_obsNoiseCov, m_hiddenStates, \
           m_arCoeffError, m_dynNoiseError, m_obsNoiseError


def arspec(arCoeffArray, dynNoiseCov, df):
    """
      Estimate the power spectrum from VAR parameter.

      Parameter
      arCoeffArray: Array of transition matrices
      dynNoiseCov:  Covariance matrix of the dynamic noise
      df:           Desired frequency spacing of the power spectrum

      Returns as first argument the power spectrum, as second the frequencies
      for which the spectrum is evaluated. The Nyquist frequency is fixed at
      pi, since an VAR has a natural sampling rate of 1 step which can be in
      application any amount of time. Hence, you have to scale the frequency
      vector by a factor, such that the last bin equals the Nyquist frequency
      of the data.

      CAVE
      Please note that the formula for the power spectrum given in the
      publication to this program is wrong. It was overlooked that in the
      multivariate case the spectrum is calculated from a matrix-valued
      formula and not simply by taking the diagonal elements and treat them
      as univariate . Thus, the correct formula for the spectrum S(w) is
        S(w) = s Q s*, with
        s(w) = inv((1-c)), with
        c(w) = sum_{o=1}^p(A[o]exp(-i o w)), where
        s* is the adjoint matrix of s,
        i is the imaginary unit,
        o indexes the order time lag of the transition matrix A
    """
    freqences = np.pi*np.linspace(0, 1, 1/df+1)
    nFreq = freqences.shape[0]

    order = len(arCoeffArray)
    dim = arCoeffArray[0].shape[0]

    m_powerSpectrum = np.zeros([nFreq, dim, dim])
    fCount = 0
    for f in freqences:
        s = 0+0j
        for i in range(0, order):
            s += arCoeffArray[i]*np.exp(-1j*f*(i+1))
        s = np.linalg.inv(np.eye(dim)-s)
        sH = np.conj(s.T)

        m_powerSpectrum[fCount, :, :] = np.abs(s.dot(dynNoiseCov).dot(sH))
        fCount += 1

    return m_powerSpectrum, freqences
