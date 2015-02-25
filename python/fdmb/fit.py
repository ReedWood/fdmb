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
libfdmb.py_emfit.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                             ct.c_int32,
                             ct.c_int32,
                             ct.c_int32,
                             ct.c_double,
                             ct.c_double,
                             ct.c_int32,
                             np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                             ]

libfdmb.py_emfit.restypes = np.int32


# Interface for libfdmb.em_arfit
def emfit(data, nData, dim, order, aThresh, pThresh, maxIter):
    # Ensure correct dtype and layout of data
    m_data = np.array(data, dtype=np.float64, order='C')
    m_arCoefficients = np.zeros((dim*order, dim*order), dtype=np.float64, order='C')

    status = libfdmb.py_emfit(m_data, nData, dim, order, aThresh, pThresh, maxIter,
                              m_arCoefficients)

    # Split coefficient matrix
    m_arCoeffArray = np.hsplit(m_arCoefficients[:dim, :], order)
    return m_arCoeffArray
