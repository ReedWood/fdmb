/*
 * Declarations of fitting functions.
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


#ifndef fdmb_fit_h
#define fdmb_fit_h


#include "global.h"


void arfit(const Matrix &x, int order, Matrix &a, Matrix &q, Matrix &v);


#endif
