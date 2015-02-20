"""@package docstring
Python wrapper for libfit.

This package defines function and return types for functions in libfit.
"""

import ctypes as ct
import numpy as np


# Load libfit and define input types
ct.cdll.LoadLibrary("/home/wmader/Methods/fdmb-refactor/build/pyinterface/libpyfit.so")
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
