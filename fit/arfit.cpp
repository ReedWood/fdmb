/*
 * Fit parameters in a VAR model.
 * Copyright (C) 2011  Yannick Linke
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


#include "global.h"
#include "fit.h"


// Optimized for low memory usage
void arfit(const Matrix &x, int order, Matrix &a, Matrix &q, Matrix &v) {
  if(order + 2 >= x.rows())
    std::cerr << "ERROR: Not enough data. Try a smaller order." << std::endl << die();

  std::vector< Matrix > acf(order + 1);

  for(int p = 0; p <= order; ++p)
    acf[p] = x.block(p, 0, x.rows() - p, x.cols()).transpose() * x.block(0, 0, x.rows() - p, x.cols());

  std::vector< std::vector< Matrix > > vACF_0(order), vACF_1(order);

  for(int row = 0; row < order; ++row) {
    vACF_0[row].resize(order);
    vACF_1[row].resize(order);

    for(int col = 0; col < order; ++col)
      if(row <= col) {
        vACF_0[row][col] = acf[col-row];
        vACF_1[row][col] = acf[col-row+1];
      } else {
        vACF_0[row][col] = acf[row-col].transpose();
        vACF_1[row][col] = acf[row-col+1].transpose();
      }
  }

  v = mcat(vACF_0);

  Matrix ACF_1 = mcat(vACF_1);
  Matrix A = ACF_1 * v.inverse();
  Matrix Q = (v - A * ACF_1.transpose()) / x.rows();

  a = A.block(0, 0, x.cols(), x.cols() * order);
  q = Q.block(0, 0, x.cols(), x.cols());

  v /= x.rows();
}
