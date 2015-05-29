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
    Fit parameters in the vector autoregressive (VAR) model.

    Parameters:
    data: Measurements, observations
    nData: Number of measurement points
    dim: Number of measured channels, dimension of the VAR
    order: Order of the VAR

    Return:
    A: Estimates of the transition matrices
    Q: Estimate of the covariance matrix nu

    The <dim>-dimensional VAR[<order]] model
       x(t) = A(1).dot(x(t-1)) + A(2).dot(x(t-2)) +
              ... + A(order).dot(x(t-n)) + nu(t)
    is fitted to <data>.
    x(t) is a <dim>-dimensional vector of which there an
    <nData> many.

    The transition matrices A(1), ..., A(order) are of
    dimension <dim>x<dim>, as is the covariance matrix Q of
    the driving noise nu. eta is assumed to be normally
    distributed.

    A is the estimate of the transition matrices returned
    as a list of length <order>. Q is the estimate of the
    covariance of the driving noise nu.

    Example:
    A, Q = fdmb.arfit(data, nData, dim, order)
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

    return m_arCoeffArray, m_noiseCovariance


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


# Interface for libfdmb.py_emfit
def emfit(data, nData, dim, order, aThresh,
          maxIter=np.int(1e-4), pThresh=1e-10, estError=False,
          initTransitionMatrix=np.nan*np.eye(1)):
    """
    Fit parameters in the state space model (SSM) using the
    expectation maximization (EM) algorithm.

    Parameters:
    data: Measurements, observations
    nData: Number of measurement points
    dim: Dimension of the VAR in the SSM
    order: Order of the VAR in the SSM
    aThresh: Convergence threshold for the EM algorithm

    Optional parameters:
    maxIter: Maximum number of EM iterations
    pThresh: Convergence threshold for fixed-point equations
    estError: 1-sigma confidence bound of parameter estimates
    initTransitionMatrix: Initial values for A (see arfit)

    Return:
    A: Estimates of the transition matrices
    Q: Estimate of the covariance matrix nu
    R: Estimate of the covariance matrix of eta
    x: Estimate of the hidden states
    AErr, QErr, RErr: 1-sigma confidence bounds of
                      parameter estimates.
                      READ FOLLOWING DOCUMENTATION!

    The state space model
      x(t) = A(1).dot(x(t-1)) + A(2).dot(x(t-2)) +
                ... + A(order).dot(x(t-n)) + nu(t)
      y(t) = C.dot(x(t)) + eta(t)
    is fitted to <data>. In the model y(t) corresponds to
    <data>. Thus, the true dynamics x(t) is not observed.
    Therefore, x(t) are called hidden states. Using the EM
    algorithm, maximum likelihood estimates of the
    parameters A, Q, and R are obtained. In order to fixed
    a gauge-invariance of the SSM, the observation matrix
    C is fixed to the identity matrix.

    The EM algorithm provides an iterative maximum
    likelihood estimator. The iteration stops if a
    convergence criterion or the maximum number of
    iteration <maxIter> is reached. Convergence is reached
    if the largest relative parameter change from one EM
    iteration to the next is below <aThresh>.

    To speed up the EM algorithm and to increase its
    accuracy assuming the process generating <data> is
    stationary on the time scale of the observation, fixed
    point equations for the gains of the Kalman filter and
    smoother are calculated also iteratively. The
    convergence of this iteration is defined by pThresh
    and defaults to 10e-10.

    Using the incomplete data likelihood, it is possible to
    obtain confidence bounds for the parameter estimates.
    If the optional parameter <estError> is set to 'True'
    these bounds are calculated. The code is there but
    needs some more testing. Therefore, the error estimates
    must be considered experimental. The documentation will
    be updated, once this shortcoming is resolved.

    Example:
      A, Q, R, x, AErr, QErr, RErr =
        fdmb.emfit(obs, nData, dim, order, aThresh, maxIter, pThresh, True)

    For a complete example please see tests/testEMFit.py.
    """
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

      Parameters:
      arCoeffArray: Array of transition matrices
      dynNoiseCov:  Covariance matrix of the dynamic noise
      df:           Frequency spacing of the power spectrum

      Return:
      ps: The power spectrum of the VAR process
      freq: Frequencies for which ps is evaluated

      Returns as first argument the power spectrum, as
      second the frequencies for which the spectrum is
      evaluated. The Nyquist frequency is fixed at pi,
      since an VAR has a natural sampling rate of 1 step
      which can in application correspond to any amount of
      time. Hence, the frequency axis must be scaled such
      that the last bin equals the Nyquist frequency of the
      analyzed data.

      CAVE
      Please note that the formula for the power spectrum
      given in the publication to this program is wrong.
      It was overlooked that in the multivariate case the
      spectrum is calculated from a matrix-valued formula
      and not simply by taking the diagonal elements and
      treat them as univariate. Thus, the correct formula
      for the spectrum S(w) is
        S(w) = s Q s*, with
        s(w) = inv((1-c)), with
        c(w) = sum_{o=1}^p(A[o]exp(-i o w)), where
        s* is the adjoint matrix of s,
        i is the imaginary unit,
        o indexes the time lag of the transition matrix A
      This implementation uses the correct formula.
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
