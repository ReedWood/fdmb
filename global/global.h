/*
 * Project wide convenience functions.
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


#ifndef fdmb_global_h
#define fdmb_global_h

#include "libraries.h"


typedef Eigen::MatrixXd Matrix;
typedef Eigen::MatrixXcd CMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MapMatrix;

const std::complex< double > iconst = std::complex< double >( 0, 1 );
const double Pi = std::atan( 1.0 ) * 4;

struct die {
    die( int code = 1 ) : code( code ) {}
    int code;
};


std::ostream & operator<<( std::ostream & stream, die const & other );

std::string trim( std::string line );
bool split( std::string line, char c, std::string& left, std::string& right );
bool split( std::string line, std::string c, std::string& left, std::string& right );
size_t replace( std::string & str, std::string const & fstr, std::string const & rstr );

double relativeChange( Matrix const & current, Matrix const & saved, double chop = 1.e-10 );

Matrix mhcat( std::vector< Matrix > const & x );
Matrix mvcat( std::vector< Matrix > const & x );
Matrix mcat( std::vector< std::vector< Matrix > > const & x );

Matrix kron( Matrix const & m1, Matrix const & m2 );
Matrix vec( Matrix const & m );
Matrix reshape( Matrix const & x, size_t n, size_t m );

Matrix pinv( Matrix const & m );

#endif
