"""@package docstring
Python wrapper for libfit.

This package defines function and return types for functions in libfit.
"""

import ctypes as ct
import numpy as np


# Load libfit and define input types
ct.cdll.LoadLibrary("libpyfit.so")
libfit = ct.CDLL("libpyfit.so")
wrp_arfit = libfit.py_arfit


# py_arfit: argtypes, restypes
wrp_arfit.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                      ct.c_int32,
                      ct.c_int32,
                      ct.c_int32,
                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                      np.ctypeslib.ndpointer(dtype=np.float64, ndim=2),
                      ]

wrp_arfit.restypes = np.int32


# Interface for wrp_arfit
def arfit(data, nData, dim, order):
    """
    Fit parameters of a VAR process.

    The parameters A1, ..., An of the VAR model
       x(t) = A1.dot(x(t-1)) + A2.dot(x(t-2)) +
              ... + An.dot(x(t-n)) + randn
    where x is a d-dimensional vector and An dXd matrix are
    fitted using Yule-Walker equations.
    """

    # Ensure correct dtype and layout of data
    m_data = np.array(data, dtype=np.float64, order='C')
    m_arCoefficients = np.zeros((dim, dim*order), dtype=np.float64, order='C')
    m_noiseCovariance = np.zeros((dim, dim), dtype=np.float64, order='C')
    m_processCovariance = np.zeros((dim*order, dim*order), dtype=np.float64, order='C')

    status = libfit.wrp_arfit(m_data, nData, dim, order,
                              m_arCoefficients, m_noiseCovariance, m_processCovariance)

    # Split coefficient matrix
    m_arCoeffArray = np.hsplit(m_arCoefficients, dim)

    return m_arCoeffArray, m_noiseCovariance, m_processCovariance
