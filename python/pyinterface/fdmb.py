"""@package docstring
Python Interface of fdmb.

Collection of all fdmb functions which interface with Python.
"""

import numpy as np
import libfit


def arfit(data, nData, dim, order):
    """
    Fit parameters of a VAR process.

    The parameters A1, ..., An of the VAR model
       x(t) = A1.dot(x(t-1)) + A2.dot(x(t-2)) +
              ... + An.dot(x(t-n)) + randn
    where x is a d-dimensional vector and A dXd matrix are
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
