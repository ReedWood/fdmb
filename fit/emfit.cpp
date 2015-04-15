/*
 * Maximum likelihood parameter estimate in the state space model.
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


#include <tbb/tbb.h>
#include "global.h"
#include "fit.h"

class ApplyKalman {
public:
    ApplyKalman( std::vector< Panel > & panels, Model const & model ) {
        init( panels, model );
    }
    
    ApplyKalman( ApplyKalman const & other, tbb::split ) {
        init( *other.panels, *other.model );
    }
    
    void init( std::vector< Panel > & panels, Model const & model ) {
        this->panels = &panels;
        this->model = &model;
        s_11 = Matrix::Zero( model.dimX, model.dimX );
        s_10 = Matrix::Zero( model.dimX, model.dimX );
        s_00 = Matrix::Zero( model.dimX, model.dimX );
        s_R = Matrix::Zero( model.dim, model.dim );
        
        if( model.input ) {
            t_11 = Matrix::Zero( model.dimX, model.dimU );
            t_01 = Matrix::Zero( model.dimX, model.dimU );
        }
    }
    
    void join( ApplyKalman const & other ) {
        s_11 += other.s_11;
        s_10 += other.s_10;
        s_00 += other.s_00;
        s_R += other.s_R;
        
        if( model->input ) {
            t_11 += other.t_11;
            t_01 += other.t_01;
        }
    }
    
    void operator()( tbb::blocked_range< size_t > const & r ) {
        for( size_t i = r.begin(); i != r.end(); ++i )
            kalman( (*panels)[i], *model );
    }

    Matrix pow( Matrix const & m, double x ) {
        Eigen::EigenSolver< Matrix > es( m );
        return (es.eigenvectors() * es.eigenvalues().array().pow( x ).matrix().asDiagonal() * es.eigenvectors().inverse()).real();
    }
    
    void kalman( Panel & panel, Model const & model ) {
        Matrix const & y = *panel.y;
        Matrix & x = *panel.x;
        Matrix const & u = *panel.u;
        Matrix const & c = model.c;
        Matrix const & k = model.k;
        Matrix const & b = model.b;
        Matrix const & ikca = model.ikca;
        Matrix const & iba = model.iba;
        Matrix const & ikcgh = model.ikcgh;
        size_t const & length = panel.length;
        
        x.block( 0, 1, model.dimX, length ) = k * y.transpose();
        if( model.input )
            x.block( 0, 1, model.dimX, length ) += ikcgh * u.transpose();
        for( size_t t = 1; t <= length; ++t )
            x.col(t) += ikca * x.col(t-1);
        x.block( 0, 0, model.dimX, length ) = iba * x.block( 0, 0, model.dimX, length );
        for( size_t t = length; t >= 1; --t )
            x.col(t-1) += b * x.col(t);
    
        s_11 += x.block( 0, 1, model.dimX, length ) * x.block( 0, 1, model.dimX, length ).transpose();
        s_10 += x.block( 0, 1, model.dimX, length ) * x.block( 0, 0, model.dimX, length ).transpose();
        s_00 += x.block( 0, 0, model.dimX, length ) * x.block( 0, 0, model.dimX, length ).transpose();
        Matrix diff = y.transpose() - c * x.block( 0, 1, model.dimX, length );
        s_R += diff * diff.transpose();
        
        if( model.input ) {
            t_11 += x.block( 0, 1, model.dimX, length ) * u;
            t_01 += x.block( 0, 0, model.dimX, length ) * u;
        }
    }
    
    Matrix s_11, s_10, s_00, s_R;
    Matrix t_11, t_01;
    
protected:
    std::vector< Panel > * panels;
    Model const * model;
};

static void iterateDARE( Model & model, double const & threshold ) {
    Matrix const & a = model.a;
    Matrix const & c = model.c;
    Matrix const & q = model.q;
    Matrix const & r = model.r;
    Matrix const & g = model.g;
    Matrix const & h = model.h;
    Matrix & p = model.p;
    Matrix & p_1 = model.p_1;
    Matrix & k = model.k;
    Matrix & b = model.b;
    Matrix & ikca = model.ikca;
    Matrix & iba = model.iba;
    Matrix & ikcgh = model.ikcgh;
    
    Matrix id = Matrix::Identity( model.dimX, model.dimX );
    
    Matrix psav;
    do {
        p_1 = a * p * a.transpose() + q;
        k = p_1 * c.transpose() * ( c * p_1 * c.transpose() + r ).inverse();
        psav = p;
        p = (id - k * c) * p_1;
		//std::cout << "Change = " << relativeChange( p, psav ) << std::endl;
    } while( relativeChange( p, psav ) > threshold );

    b = p * a.transpose() * p_1.inverse();
    ikca = (id - k * c) * a;
    iba = id - b * a;
    if( model.input )
        ikcgh = (id - k * c) * g - k * h;
}

static void calcPSums( ApplyKalman & sum, Model const & model, size_t const & totalLength ) {
    Matrix const & a = model.a;
    Matrix const & c = model.c;
    Matrix const & p = model.p;
    Matrix const & p_1 = model.p_1;
    Matrix const & b = model.b;
    
    Matrix W = ( Matrix::Identity( model.dimX * model.dimX, model.dimX * model.dimX ) - kron( b, b ) ).inverse();

    sum.s_11 = sum.s_00 = reshape( W * vec( p - b * p_1 * b.transpose() ), model.dimX, model.dimX );
    sum.s_10 = reshape( W * vec( ( Matrix::Identity( model.dimX, model.dimX ) - b * a ) * p * b.transpose() ), model.dimX, model.dimX );
    sum.s_11 *= totalLength;
    sum.s_10 *= totalLength;
    sum.s_00 *= totalLength;
    sum.s_R = c * sum.s_11 * c.transpose();
}

static void calcPSums( ApplyKalman & sum, Model const & model, size_t const & totalLength, double const & threshold ) {
    Matrix const & a = model.a;
    Matrix const & c = model.c;
    Matrix const & p = model.p;
    Matrix const & p_1 = model.p_1;
    Matrix const & b = model.b;
    Matrix & s_00 = sum.s_00;
    Matrix & s_10 = sum.s_10;
    Matrix & s_11 = sum.s_11;
    Matrix & s_R = sum.s_R;
    
    s_11 = p - b * p_1 * b.transpose();
    s_10 = ( Matrix::Identity( model.dimX, model.dimX ) - b * a ) * p * b.transpose();
    Matrix s_11sav, s_10sav;

    Matrix z = s_11;
    do {
        s_11sav = s_11;
        z = b * z * b.transpose();
        s_11 += z;
    } while( relativeChange( s_11, s_11sav ) > threshold );
    s_11 *= totalLength;
    //std::cout << i << std::endl << std::flush;
    
    z = s_10;
    do {
        s_10sav = s_10;
        z = b * z * b.transpose();
        s_10 += z;
    } while( relativeChange( s_10, s_10sav ) > threshold );
    s_10 *= totalLength;

    s_00 = s_11;
    s_R = c * s_11 * c.transpose();
}

static double update( Model & model, ApplyKalman const & sum, size_t const & totalLength ) {
    Matrix & a = model.a;
    Matrix const & c = model.c;
    Matrix & q = model.q;
    Matrix & r = model.r;
    Matrix & g = model.g;
    Matrix & h = model.h;
    Matrix & asav = model.asav;
    Matrix const & s_00 = sum.s_00;
    Matrix const & s_10 = sum.s_10;
    Matrix const & s_11 = sum.s_11;
    Matrix const & s_R = sum.s_R;
    Matrix const & t_01 = sum.t_01;
    Matrix const & t_11 = sum.t_11;
    Matrix const & t_u = model.t_u;
    Matrix const & t_y = model.t_y;
    Matrix t_uInv;
    
    if( model.input )
        t_uInv = t_u.inverse();
    
    asav = a;
    if( model.input ) {
        Matrix u = (s_00 - t_01 * t_uInv * t_01.transpose()).inverse();
        Matrix v = -u * t_01 * t_uInv;
        Matrix w = t_uInv + t_uInv * t_01.transpose() * u * t_01 * t_uInv;
        a = s_10 * u + t_11 * v.transpose();
        g = s_10 * v + t_11 * w;
        g.block( model.dim, 0, (model.order - 1) * model.dim, model.dimU ).setZero();
    } else
        a = sum.s_10 * sum.s_00.inverse();
    a.block( model.dim, 0, (model.order - 1) * model.dim, model.dimX ).setIdentity();

    Matrix qUpdate = sum.s_11 - sum.s_10 * a.transpose() - a * sum.s_10.transpose() + a * sum.s_00 * a.transpose();
    if( model.input )
        qUpdate += a * t_01 * g.transpose() + g * t_01.transpose() * a.transpose() + g * t_u * g.transpose() - t_11 * g.transpose() - g * t_11.transpose();
    
    q.setZero();
    q.topLeftCorner( model.dim, model.dim ) = 1. / (totalLength - 1) * qUpdate.topLeftCorner( model.dim, model.dim );
    q = (q + q.transpose().eval()) / 2;
    
    if( model.input && model.est_h )
        h = (t_y - c * t_11) * t_uInv;
    
    Matrix rUpdate = s_R;
    if( model.input && model.est_h )
        rUpdate += c * t_11 * h.transpose() + h * t_11.transpose() * c.transpose() + h * t_u * h.transpose() - t_y * h.transpose() - h * t_y.transpose();

    r = rUpdate;
    r /= totalLength - 1;
    r = (r + r.transpose().eval()) / 2;

    //model.v = sum.s_00 / totalLength;

    return relativeChange( a, asav );
}

static void setLogFiles( Info & info, Model const & model ) {
    std::time_t rawtime;
    std::tm* timeinfo;
    char curTime[7], curDate[9];
    
    std::time( &rawtime );
    timeinfo = std::localtime( &rawtime );
    std::strftime( curTime, 7, "%H%M%S", timeinfo );
    std::strftime( curDate, 9, "%Y%m%d", timeinfo );
    
    std::string logFilesPat = info.logFilesPat;
    
    replace( logFilesPat, "%d", curDate );
    replace( logFilesPat, "%t", curTime );
    
    if( logFilesPat.find( "%p" ) == std::string::npos )
        logFilesPat = "par%p_" + logFilesPat;
    
    logFilesPat = info.logPath + "/" + logFilesPat;
    
    info.logFiles["A"] = logFilesPat;
    info.logFiles["Q"] = logFilesPat;
    info.logFiles["R"] = logFilesPat;
    info.logFiles["V"] = logFilesPat;
    info.logFiles["X0"] = logFilesPat;
    info.logFiles["X"] = logFilesPat;
    
    if( model.input ) {
        info.logFiles["G"] = logFilesPat;
        if( model.est_h )
            info.logFiles["H"] = logFilesPat;
    }
    
    for( std::map< std::string, std::string >::iterator i = info.logFiles.begin(); i != info.logFiles.end(); ++i ) {
        replace( i->second, "%p", i->first );
        std::ofstream file( i->second.c_str() );
        if( !file )
            std::cerr << "ERROR: Cannot open log file '" << i->second << "'." << std::endl << die();
    }
}

static void checkInput( std::vector< Panel > & panels, Model & model, Info & info ) {
    if( panels.size() == 0 )
        std::cerr << "ERROR: No panels passed to emfit." << std::endl << die();
    
    model.dim = panels[0].y->cols();
    model.dimX = model.dim * model.order;
    if( model.input )
        model.dimU = panels[0].u->cols();
    info.totalLength = 0;
    for( std::vector< Panel >::iterator panel = panels.begin(); panel != panels.end(); ++panel ) {
        panel->length = panel->y->rows();
        Matrix x0 = *panel->x;
        if( x0.cols() == 0 && x0.rows() == 0 ) {
            x0 = Matrix::Zero( model.dimX, 1 );
            x0.block( 0, 0, model.dim, 1 ) = panel->y->row(0).transpose();
        }
        if( x0.rows() == 1 )
            x0.transposeInPlace();
        if( x0.cols() != 1 || x0.rows() != model.dimX )
            std::cerr << "ERROR: Dimension mismatch of x0 in panel " << (panel - panels.begin() + 1) << " in emfit."
                << std::endl << die();
        panel->x->resize( model.dimX, panel->length + 1 );
        panel->x->col(0) = x0;
        
        info.totalLength += panel->length;
        
        if( panel->y->cols() != model.dim )
            std::cerr << "ERROR: Dimension mismatch of observation " << (panel - panels.begin() + 1) << "in emfit." << std::endl << die();
        if( model.input && ( panel->u->cols() != model.dimU || panel->u->rows() != panel->y->rows() ) )
            std::cerr << "ERROR: Dimension mismatch of stimuli " << (panel - panels.begin() + 1) << " in emfit." << std::endl << die();
    }
    
    if( model.a.rows() == 0 || model.a.cols() == 0 ) {
        model.a = Matrix::Zero( model.dimX, model.dimX );
        model.a.block( 0, 0, model.dim, model.dim ) = .8 * Matrix::Identity( model.dim, model.dim );
        model.a.block( model.dim, 0, (model.order - 1) * model.dim, model.dimX ).setIdentity();
    } else if( model.a.rows() == model.dim && model.a.cols() == model.dimX ) {
        model.a.conservativeResize( model.dimX, Eigen::NoChange );
        model.a.block( model.dim, 0, (model.order - 1) * model.dim, model.dimX ).setIdentity();
	} else if( model.a.rows() == model.dimX && model.a.cols() == model.dimX ) {
        model.a.block( model.dim, 0, (model.order - 1) * model.dim, model.dimX ).setIdentity();
    } else //if( model.a.rows() != model.dimX || model.a.cols() != model.dimX ) {
        std::cerr << "ERROR: Dimension mismatch of matrix A in emfit. " << std::endl << die();
    
    if( model.c.rows() == 0 || model.c.cols() == 0 )
        model.c = Matrix::Identity( model.dim, model.dimX );
    else if( model.c.rows() == model.dim && model.c.cols() == model.dim ) {
        model.c.conservativeResize( Eigen::NoChange, model.dimX );
        model.c.block( 0, model.dim, model.dim, (model.order - 1) * model.dim ).setIdentity();
    } else //if( model.c.rows() != model.dim || model.c.cols() != model.dimX ) {
        std::cerr << "ERROR: Dimension mismatch of matrix C in emfit. " << std::endl << die();
    
    if( model.q.rows() == 0 || model.q.cols() == 0 ) {
        model.q = Matrix::Zero( model.dimX, model.dimX );
        model.q.block( 0, 0, model.dim, model.dim ).setIdentity();
    } else if( model.q.rows() == model.dim && model.q.cols() == model.dim ) {
        Matrix qtmp = model.q;
        model.q = Matrix::Zero( model.dimX, model.dimX );
        model.q.block( 0, 0, model.dim, model.dim ) = qtmp;
	} else if( model.q.rows() == model.dimX && model.q.cols() == model.dimX ) {
        Matrix qtmp = model.q.block( 0, 0, model.dim, model.dim );
        model.q = Matrix::Zero( model.dimX, model.dimX );
        model.q.block( 0, 0, model.dim, model.dim ) = qtmp;		
    } else //if( model.q.rows() != model.dimX || model.q.cols() != model.dimX ) {
        std::cerr << "ERROR: Dimension mismatch of matrix Q in emfit. " << std::endl << die();
    
    if( model.r.rows() == 0 || model.r.cols() == 0 )
        model.r = Matrix::Identity( model.dim, model.dim );
    else if( model.r.rows() != model.dim || model.r.cols() != model.dim )
        std::cerr << "ERROR: Dimension mismatch of matrix R in emfit. " << std::endl << die();
    
    if( model.input ) {
        if( model.g.rows() == 0 || model.g.cols() == 0 ) {
            model.g = Matrix::Zero( model.dimX, model.dimU );
            model.g.block( 0, 0, model.dim, model.dimU ).setIdentity();
        } else if( model.g.rows() == model.dim && model.g.cols() == model.dimU ) {
            model.g.conservativeResize( model.dimX, Eigen::NoChange );
            model.g.block( model.dim, 0, (model.order - 1) * model.dim, model.dimU ).setZero();
        } else //if( model.g.rows() != model.dimX || model.g.cols() != model.dimU ) {
            std::cerr << "ERROR: Dimension mismatch of matrix G in emfit. " << std::endl << die();
       
        if( model.h.rows() == 0 || model.h.cols() == 0 )
            if( model.est_h )
                model.h = Matrix::Identity( model.dim, model.dimU );
            else
                model.h = Matrix::Zero( model.dim, model.dimU );
        else if( model.h.rows() != model.dim || model.h.cols() != model.dimU )
            std::cerr << "ERROR: Dimension mismatch of matrix H in emfit. " << std::endl << die();
    
        model.t_u = Matrix::Zero( model.dimU, model.dimU );
        model.t_y = Matrix::Zero( model.dim, model.dimU );
        for( size_t i = 0; i < panels.size(); ++i ) {
            model.t_u += panels[i].u->transpose() * *panels[i].u;
            model.t_y += panels[i].y->transpose() * *panels[i].u;
        }
    }
    
    model.p = Matrix::Identity( model.dimX, model.dimX );
    
    model.asav = Matrix::Constant( model.dimX, model.dimX, NAN );
    
    if( info.log )
        mkdir( info.logPath.c_str(), 0777 );
    
    setLogFiles( info, model );
    
    info.started = info.checkpoint = std::clock();
}

static void log( std::vector< Panel > & panels, Model & model, Info & info ) {
    std::clock_t now = std::clock();
    
    std::ofstream logA( info.logFiles["A"].c_str(), std::ofstream::app ),
        logQ( info.logFiles["Q"].c_str(), std::ofstream::app ),
        logR( info.logFiles["R"].c_str(), std::ofstream::app ),
        logV( info.logFiles["V"].c_str(), std::ofstream::app ),
        logX0( info.logFiles["X0"].c_str(), std::ofstream::app ),
        logX( info.logFiles["X"].c_str() );
    
    logA << "Iteration: " << info.iterations << ", convergence: " << relativeChange( model.a, model.asav ) << "." << std::endl;
    logA << "Time elapsed: total: " << double( now - info.started ) / CLOCKS_PER_SEC
        << "s, last " << info.logStep << " steps: " << double( now - info.checkpoint ) / CLOCKS_PER_SEC
        << "s." << std::endl;
    logA << model.a.block( 0, 0, model.dim, model.dimX ) << std::endl << std::endl;
    logA.close();
    logQ << "Iteration: " << info.iterations << std::endl;
    logQ << model.q.block( 0, 0, model.dim, model.dim ) << std::endl << std::endl;
    logQ.close();
    logR << "Iteration: " << info.iterations << std::endl;
    logR << model.r << std::endl << std::endl;
    logR.close();
    logV << "Iteration: " << info.iterations << std::endl;
    logV << model.v << std::endl << std::endl;
    logV.close();
    
    if( model.input ) {
        std::ofstream logG( info.logFiles["G"].c_str(), std::ofstream::app ), logH;
        
        logG << "Iteration: " << info.iterations << std::endl;
        logG << model.g.block( 0, 0, model.dim, model.dimU ) << std::endl << std::endl;
        logG.close();
        
        if( model.est_h ) {
            logH.open( info.logFiles["H"].c_str(), std::ofstream::app );
            
            logH << "Iteration: " << info.iterations << std::endl;
            logH << model.h << std::endl << std::endl;
            logH.close();
        }
    }
    
    for( std::vector< Panel >::iterator panel = panels.begin(); panel != panels.end(); ++panel ) {
        logX0 << "Iteration: " << info.iterations << std::endl;
        logX0 << panel->x->col( 0 ).transpose() << std::endl << std::endl;
        logX << panel->x->block( 0, 1, model.dim, panel->length ).transpose() << std::endl << std::endl;
    }
    logX.close();
    
    info.checkpoint = now;
}

void emfit( std::vector< Panel > & panels, Model & model, Info & info ) {
    //std::cout << "check input" << std::endl << std::flush;
    checkInput( panels, model, info );
    //std::cout << "go" << std::endl << std::flush;
    std::clock_t start, end;
    for( size_t i = 0; i < info.maxIter; ++i ) {
        //std::cout << "start iteration" << std::endl << std::flush;
        info.iterations = i;
        if( info.log && i % info.logStep == 0 )
            log( panels, model, info );
        
        ApplyKalman sum( panels, model );
        
        //std::cout << "iterate dare..." << std::flush;
        start = std::clock();
        iterateDARE( model, info.pThresh );
        end = std::clock();
        //std::cout << "done in " << double( end - start ) / CLOCKS_PER_SEC << "s." << std::endl << "calculate p sums..." << std::flush;
        start = std::clock();
        calcPSums( sum, model, info.totalLength, info.pThresh );
        end = std::clock();
        
        //std::cout << "done in " << double( end - start ) / CLOCKS_PER_SEC << "s." << std::endl << "kalman..." << std::flush;
        start = std::clock();
        tbb::parallel_reduce( tbb::blocked_range< size_t >( 0, panels.size() ), sum );
        end = std::clock();
        //std::cout << "done in " << double( end - start ) / CLOCKS_PER_SEC << "s." << std::endl << std::flush;

        if( update( model, sum, info.totalLength ) < info.aThresh ) {
            //model.v = sum.s_00 / info.totalLength; //it was not designed that s_00 is returned
            break;
        }
    }
    log( panels, model, info );
}