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
  //FIXME Complete function documentation
  /**
   * @brief Fit parameters in an AR model.
   * 
   * The algorithm implements the Yule-Walker equations to estimate the parameters of an AR[p] model
   * from measurements.
   * 
   * 
   * @param data Input data, measurements
   * @param dim Dimensions of data as integer array
   * @param order Order p of the AR[p] model
   * @return void
   */
  int py_arfit(const double *data,
               const int nData,
               const int dim,
               const int order,
               double *arCoefficients,
               double *drvNoise,
               double *autoCov
              );
  
  int py_emfit(const double *data,
               const int nData,
               const int dim,
               const int order,
               const double aThresh,
               const double pThresh,
               const int maxIter,
               double *arCoefficients
              );
}
