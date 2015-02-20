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


#include "global.h"


std::ostream & operator<<( std::ostream & stream, die const & other ) {
    stream << std::flush;
    std::exit( other.code );
    return stream;
}

std::string trim( std::string line ) {
    size_t left = line.find_first_not_of( " \t\r\n" ), right = line.find_last_not_of( " \t\r\n" );
    if( left == std::string::npos || right == std::string::npos )
        return "";
    return line.substr( left, right - left + 1 );
}

bool split( std::string line, char c, std::string& left, std::string& right ) {
    size_t pos = line.find( c );
    if( pos == std::string::npos )
        return false;
    left = trim( line.substr( 0, pos ) );
    right = trim( line.substr( pos + 1 ) );
    return true;
}

bool split( std::string line, std::string c, std::string& left, std::string& right ) {
    size_t pos = line.find( c );
    if( pos == std::string::npos )
        return false;
    left = trim( line.substr( 0, pos ) );
    right = trim( line.substr( pos + c.length() ) );
    return true;
}

size_t replace( std::string & str, std::string const & fstr, std::string const & rstr ) {
    size_t pos, i = 0;
    while( (pos = str.find( fstr )) != std::string::npos ) {
        str.replace( pos, fstr.length(), rstr );
        ++i;
    }
    return i;
}

double relativeChange( Matrix const & current, Matrix const & saved, double chop ) {
    return ( current.array().abs() < chop ).select( 0, ( current - saved ).cwiseQuotient( current ) ).cwiseAbs().maxCoeff();
}

Matrix mhcat( std::vector< Matrix > const & x ) {
    if( x.size() == 0 )
        return Matrix( 0, 0 );
    
    size_t rows = x[0].rows();
    size_t cols = x[0].cols();
    for( size_t i = 1; i < x.size(); ++i ) {
        if( x[i].rows() != rows )
            std::cerr << "ERROR: Dimension mismatch in 'mhcat'." << std::endl << die();
        cols += x[i].cols();
    }
    
    Matrix m( rows, cols );
    size_t col = 0;
    for( size_t i = 0; i < x.size(); ++i ) {
        m.block( 0, col, rows, x[i].cols() ) = x[i];
        col += x[i].cols();
    }
    return m;
}

Matrix mvcat( std::vector< Matrix > const & x ) {
    if( x.size() == 0 )
        return Matrix( 0, 0 );
    
    size_t rows = x[0].rows();
    size_t cols = x[0].cols();
    for( size_t i = 1; i < x.size(); ++i ) {
        if( x[i].cols() != cols )
            std::cerr << "ERROR: Dimension mismatch in 'mvcat'." << std::endl << die();
        rows += x[i].rows();
    }
    
    Matrix m( rows, cols );
    size_t row = 0;
    for( size_t i = 0; i < x.size(); ++i ) {
        m.block( row, 0, x[i].rows(), cols ) = x[i];
        row += x[i].rows();
    }
    return m;
}

Matrix mcat( std::vector< std::vector< Matrix > > const & x ) {
    std::vector< Matrix > h( x.size() );
    for( size_t i = 0; i < x.size(); ++i )
        h[i] = mhcat( x[i] );
    return mvcat( h );
}

Matrix kron( Matrix const & m1, Matrix const & m2 ) {
    Matrix out( m1.rows() * m2.rows(), m1.cols() * m2.cols() );
    
    for( size_t i = 0; i < m1.cols(); ++i )
        for( size_t j = 0; j < m1.rows(); ++j )
            out.block( i * m2.rows(), j * m2.cols(), m2.rows(), m2.cols() ) = m1( i, j ) * m2;
    
    return out;
}

Matrix vec( Matrix const & m ) {
    return Eigen::Map< Matrix const >( m.data(), m.rows() * m.cols(), 1 );
}

Matrix reshape( Matrix const & x, size_t n, size_t m ) {
    return Eigen::Map< Matrix const >( x.data(), n, m );
}

Matrix pinv( Matrix const & m ) {
    double const tol = 1e-6;
    Eigen::JacobiSVD< Matrix > svd( m, Eigen::ComputeThinU | Eigen::ComputeThinV );
    Matrix sv = svd.singularValues();
    size_t n = m.cols() < m.rows() ? m.cols() : m.rows();
    for( size_t i = 0; i < n; ++i )
        if ( sv(i) > tol )
            sv(i) = 1.0 / sv(i);
        else
            sv(i) = 0;
    return svd.matrixV() * sv.asDiagonal() * svd.matrixU().transpose();
}
