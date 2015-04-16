/*
 * C naming export of fitting functions.
 * Copyright (C) 2015  Wolfgang Mader <Wolfgang.Mader@fdm.uni-freiburg.de>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


/**
 * @file pyFit.h
 * @brief Export functions of fit.h to python.
 */


#include "global.h"
#include "fit.h"

extern "C" {
  /**
   * @brief Fit parameters in an AR model.
   * 
   * The algorithm relies on the Yule-Walker equations to estimate the
   * parameters of an AR[p] model from measurements.
   * 
   * 
   * @param data Input data, measurements
   * @param dim Dimensions of data as integer array
   * @param order Order p of the AR[p] model
   * @param arCoefficients Estimate of the transition matrices
   * @param drvNoise Estimate of the dynamic noise in the VAR model
   * @param autoCov Autocovariance of the vector process
   * @return int Useless status indicator, always 0
   */
  int py_arfit(const double *data,
               const int nData,
               const int dim,
               const int order,
               double *arCoefficients,
               double *drvNoise,
               double *autoCov
              );
  
  /**
   * @brief Maximum-likelihood estimation in the state-space model (SSM)
   * 
   * @param data Input data, measurements
   * @param nData Length of the measurements, number of sample points
   * @param dim Dimension of data
   * @param order Order p of the VAR[p] used in the SSM
   * @param aThresh Threshold for the relative change of parameters defining convergence
   * @param pThresh Threshold defining the converge of the filter to the steady state
   * @param maxIter Maximum number of iterations
   * @param arCoefficients Estimates of the transition matrices of the VAR[p]
   * @param dynNoiseCov Estimate of the covariance matrix of the dynamic noise
   * @param obsNoiseCov Estimate of the covariance matrix of the observational noise
   * @param hiddenStates Estimate of the hidden states of the SSM
   * @return int Useless status indicator, always 0
   * 
   * The algorithm implements a maximum-likelihood estimator for the parameters
   * A, Q, and R in the state-space model (SSM). It is
   *    A: Transition matrices of the VAR[p] used in the SSM
   *    Q: Covariance matrix of the dynamic noise of the VAR[p]
   *    R: Covariance matrix of the observational noise
   * In addition, an estimate of the hidden states of the VAR are returned.
   * 
   * The estimator is based on the expectation-maximization (EM) algorithm and 
   * is thus iterative. In fact, two iterations appear in this optimized
   * version, such that two threshold must be provided.
   * 
   * aThresh
   * The relative change in the transition matrix must go below this threshold
   * in order for the estimator to converge. This threshold defines the global
   * iteration.
   * 
   * pThresh
   * In this optimized version of the EM algorithm, the filter is brought to
   * steady-state such that the covariance matrices of the prediction error and
   * the Kalman gain have stabilized. To this end, they are iterated until they
   * relative change of the covariance matrix goes below pThresh.
   */
  int py_emfit(const double *data,
               const int nData,
               const int dim,
               const int order,
               const double aThresh,
               const double pThresh,
               const int maxIter,
               double *arCoefficients,
               double *dynNoiseCov,
               double *obsNoiseCov,
               double *hiddenStates,
               double *estimationError,
               const bool estError,
               const double *initTransitionMatrix
              );
}
