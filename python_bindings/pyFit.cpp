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


#include "pyFit.h"
#include "fit.h"


int py_arfit(const double *data,
             const int nData,
             const int dim,
             const int order,
             double *arCoefficients,
             double *drvNoise,
             double *autoCov
            )
{
  Eigen::Map<const MapMatrix> dataMap(data, nData, dim);


  Matrix coefficientMatrix;
  Matrix noiseMatrix;
  Matrix covarianceMatrix;

  arfit(dataMap, order, coefficientMatrix, noiseMatrix, covarianceMatrix);


  Eigen::Map<MapMatrix> coefficientMap(arCoefficients, dim, dim*order);
  Eigen::Map<MapMatrix> noiseMap(drvNoise, dim, dim);
  Eigen::Map<MapMatrix> covarianceMap(autoCov, dim*order, dim*order);

  coefficientMap = coefficientMatrix;
  noiseMap = noiseMatrix;
  covarianceMap = covarianceMatrix;

  return 0;
}


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
             double *hiddenStates
            )
{
  Eigen::Map<const MapMatrix> dataMap(data, nData, dim);

  Matrix y = dataMap;
  Matrix x;

  std::vector<Panel> panels(1);
  panels[0].y = &y;
  panels[0].x = &x;
  
  Model model;
  model.order = order;
  model.input = false;
  model.est_h = false;

  Info info;
  info.aThresh = aThresh;
  info.pThresh = pThresh;
  info.maxIter = maxIter;
  info.logPath = "/tmp/fdmb/";
  info.log = true;

  emfit(panels, model, info);
    
  Eigen::Map<MapMatrix> coefficientMap(arCoefficients, dim*order, dim*order);
  coefficientMap = model.a;
  
  Eigen::Map<MapMatrix> dynNoiseCovMap(dynNoiseCov, dim*order, dim*order);
  dynNoiseCovMap = model.q;
  
  Eigen::Map<MapMatrix> obsNoiseCovMap(obsNoiseCov, dim, dim);
  obsNoiseCovMap = model.r;
  
  Eigen::Map<MapMatrix> hiddenStatesMap(hiddenStates, dim*order, nData+1);
  hiddenStatesMap = *(panels[0].x);
  
  return 0;
}


