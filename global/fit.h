/*
 * Declarations of fitting functions.
 * Copyright (C) 2011  Yannick Linke
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


#ifndef fdmb_fit_h
#define fdmb_fit_h


#include <ctime>
#include "global.h"


struct Panel {
    size_t length;
    Matrix * y, * x ;
    Matrix * u;
};


struct Model {
    size_t order, ma_order, dim, dimX, dimU;
    Matrix a, m, c, q, r, p;
    Matrix g, h;
    Matrix p_1, k, b, t, tqtt, ts, ikca, iba, ikcgh;
    Matrix asav;
    
    Matrix v;
    
    bool input, est_h;
    Matrix t_u, t_y;
};


struct Info {
    double aThresh, pThresh;
    size_t maxIter;
    size_t totalLength;
    
    bool log;
    size_t logStep;
    std::string logPath;
    std::string logFilesPat;
    std::map< std::string, std::string > logFiles;
    
    size_t iterations;
    std::clock_t started, checkpoint;
};


void arfit(const Matrix &x, int order, Matrix &a, Matrix &q, Matrix &v);
void emfit(std::vector<Panel> & panels, Model & model, Info & info);

#endif
