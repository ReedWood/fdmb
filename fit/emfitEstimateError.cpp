/*
 * Covariance of the maximum-likelihood estimates obtained from emfit.
 * Copyright (C) 2013  Linda Sommerlade
 * Copyright (C) 2015  Wolfgang Mader <Wolfgang.Mader@fdm.uni-freiburg.de>
 * This file is mostly a copy of V_analytically.cpp authored by Linde Sommerlade
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


class calc_dd_incompl_likelihood {
public:
    calc_dd_incompl_likelihood( std::vector< Panel > & panels, Model & model) {
        initV( panels, model );
    }

    calc_dd_incompl_likelihood( calc_dd_incompl_likelihood const & other, tbb::split ) {
        initV( *other.panels, *other.model );
    }

    void initV( std::vector< Panel > & panels, Model & model ) {
        this->panels = &panels;
        this->model = &model;
    }

    void join( calc_dd_incompl_likelihood const & other ) {

    }

    void operator()( tbb::blocked_range< size_t > const & r ) {
        for( size_t i = r.begin(); i != r.end(); ++i )
            dd_incompl_likelihood( (*panels)[i], *model );
    }

    Matrix pow( Matrix const & m, double x ) {
        Eigen::EigenSolver< Matrix > es( m );
        return (es.eigenvectors() * es.eigenvalues().array().pow( x ).matrix().asDiagonal() * es.eigenvectors().inverse()).real();
    }

    void dd_incompl_likelihood( Panel & panel, Model & model) {
        Matrix const & y = *panel.y;
        Matrix const & u = *panel.u;
        Matrix const & a = model.a;
    Matrix const & q = model.q;
    Matrix const & r = model.r;
    //Matrix const r = Matrix::Zero(model.r.rows(),model.r.cols());
    Matrix const & c = model.c;
    Matrix const & g = model.g; //bisher nicht verwendet
    Matrix const & h = model.h; //bisher nicht verwendet
    size_t const & length = panel.length;
    size_t dimX = model.dimX;
    size_t order = model.order;
    Matrix id = Matrix::Identity( dimX, dimX );

    size_t dim = model.dim;
    size_t dimA = dimX*floor(dimX/order);
    size_t dimQ = floor(dimX/order)*floor(dimX/order);
    size_t dimR = dim*dim;
    size_t dimSum = dimA+dimQ+dimR;


    Matrix likeE;
    Matrix d1likeE;
    Matrix d2likeE;
    Matrix ddlikeE;
    Matrix d1likeEQ;
    Matrix d2likeEQ;
    Matrix d1likeER;
    Matrix d2likeER;
    Matrix ddlikeEQQ;
    Matrix ddlikeEQR;
    Matrix ddlikeERQ;
    Matrix ddlikeERR;
    Matrix ddlikeEAQ;
    Matrix ddlikeEQA;
    Matrix ddlikeEAR;
    Matrix ddlikeERA;
    Matrix inverselikeS;
    Matrix likeS;
    Matrix d1likeS;
    Matrix d2likeS;
    Matrix ddlikeS;
    Matrix d1likeSQ;
    Matrix d2likeSQ;
    Matrix d1likeSR;
    Matrix d2likeSR;
    Matrix ddlikeSQQ;
    Matrix ddlikeSQR;
    Matrix ddlikeSRQ;
    Matrix ddlikeSRR;
    Matrix ddlikeSAQ;
    Matrix ddlikeSQA;
    Matrix ddlikeSAR;
    Matrix ddlikeSRA;
    Matrix logLiketmp1;
    Matrix logLiketmp2 = Matrix::Zero(1, 1);
    double logLike1 = 0.0;
    double logLike2 = 0.0;

    V = Matrix::Zero(dimSum,dimSum);

    Matrix kdenominator = Matrix::Zero(r.rows(),r.rows());

    Matrix eiejT;
    Matrix ekelT;

    size_t V_ir;
    size_t V_ic;

    for (size_t iorder1 = 0; iorder1 < order; ++iorder1) {
      for (size_t ir1=0; ir1 < floor(dimX/order); ++ir1) {
        for (size_t ic1=0; ic1 < floor(dimX/order); ++ic1) {
          for (size_t iorder2 = 0; iorder2 < order; ++iorder2) {
            for (size_t ir2=0; ir2 < floor(dimX/order); ++ir2) {
              for (size_t ic2=0; ic2 < floor(dimX/order); ++ic2) {

                V_ir = iorder1*(floor(dimX/order)*floor(dimX/order)) + ir1*(floor(dimX/order)) + ic1;
                V_ic = iorder2*(floor(dimX/order)*floor(dimX/order)) + ir2*(floor(dimX/order)) + ic2;
                if (V_ic <= V_ir)
                {

                  Matrix k = model.k;
                  Matrix p = model.p;
                  Matrix x = *panel.x;

                  Matrix p_1 = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix d1p_1 = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix d2p_1 = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix ddp_1 = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix d1p_1Q = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix d2p_1Q = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix d1p_1R = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix d2p_1R = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix ddp_1QQ = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix ddp_1QR = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix ddp_1RQ = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix ddp_1RR = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix ddp_1AQ = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix ddp_1QA = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix ddp_1AR = Matrix::Zero(model.p_1.rows(),model.p_1.cols());
                  Matrix ddp_1RA = Matrix::Zero(model.p_1.rows(),model.p_1.cols());

                  Matrix x_1 = Matrix::Zero(x.rows(),1);
                  Matrix d1x_1 = Matrix::Zero(x.rows(),1);
                  Matrix d2x_1 = Matrix::Zero(x.rows(),1);
                  Matrix ddx_1 = Matrix::Zero(x.rows(),1);
                  Matrix d1x_1Q = Matrix::Zero(x.rows(),1);
                  Matrix d2x_1Q = Matrix::Zero(x.rows(),1);
                  Matrix d1x_1R = Matrix::Zero(x.rows(),1);
                  Matrix d2x_1R = Matrix::Zero(x.rows(),1);
                  Matrix ddx_1QQ = Matrix::Zero(x.rows(),1);
                  Matrix ddx_1QR = Matrix::Zero(x.rows(),1);
                  Matrix ddx_1RQ = Matrix::Zero(x.rows(),1);
                  Matrix ddx_1RR = Matrix::Zero(x.rows(),1);
                  Matrix ddx_1AQ = Matrix::Zero(x.rows(),1);
                  Matrix ddx_1QA = Matrix::Zero(x.rows(),1);
                  Matrix ddx_1AR = Matrix::Zero(x.rows(),1);
                  Matrix ddx_1RA = Matrix::Zero(x.rows(),1);

                  Matrix d1x = Matrix::Zero(x.rows(),x.cols());
                  Matrix d2x = Matrix::Zero(x.rows(),x.cols());
                  Matrix ddx = Matrix::Zero(x.rows(),x.cols());
                  Matrix d1xQ = Matrix::Zero(x.rows(),x.cols());
                  Matrix d2xQ = Matrix::Zero(x.rows(),x.cols());
                  Matrix d1xR = Matrix::Zero(x.rows(),x.cols());
                  Matrix d2xR = Matrix::Zero(x.rows(),x.cols());
                  Matrix ddxQQ = Matrix::Zero(x.rows(),x.cols());
                  Matrix ddxQR = Matrix::Zero(x.rows(),x.cols());
                  Matrix ddxRQ = Matrix::Zero(x.rows(),x.cols());
                  Matrix ddxRR = Matrix::Zero(x.rows(),x.cols());
                  Matrix ddxAQ = Matrix::Zero(x.rows(),x.cols());
                  Matrix ddxQA = Matrix::Zero(x.rows(),x.cols());
                  Matrix ddxAR = Matrix::Zero(x.rows(),x.cols());
                  Matrix ddxRA = Matrix::Zero(x.rows(),x.cols());

                  Matrix pt_1 = Matrix::Zero(p.rows(),p.cols());
                  Matrix d1pt_1 = Matrix::Zero(p.rows(),p.cols());
                  Matrix d2pt_1 = Matrix::Zero(p.rows(),p.cols());
                  Matrix d1pt_1Q = Matrix::Zero(p.rows(),p.cols());
                  Matrix d2pt_1Q = Matrix::Zero(p.rows(),p.cols());
                  Matrix d1pt_1R = Matrix::Zero(p.rows(),p.cols());
                  Matrix d2pt_1R = Matrix::Zero(p.rows(),p.cols());
                  Matrix d1p = Matrix::Zero(p.rows(),p.cols());
                  Matrix d2p = Matrix::Zero(p.rows(),p.cols());
                  Matrix ddp = Matrix::Zero(p.rows(),p.cols());
                  Matrix d1pQ = Matrix::Zero(p.rows(),p.cols());
                  Matrix d2pQ = Matrix::Zero(p.rows(),p.cols());
                  Matrix d1pR = Matrix::Zero(p.rows(),p.cols());
                  Matrix d2pR = Matrix::Zero(p.rows(),p.cols());
                  Matrix ddpQQ = Matrix::Zero(p.rows(),p.cols());
                  Matrix ddpQR = Matrix::Zero(p.rows(),p.cols());
                  Matrix ddpRQ = Matrix::Zero(p.rows(),p.cols());
                  Matrix ddpRR = Matrix::Zero(p.rows(),p.cols());
                  Matrix ddpAQ = Matrix::Zero(p.rows(),p.cols());
                  Matrix ddpQA = Matrix::Zero(p.rows(),p.cols());
                  Matrix ddpAR = Matrix::Zero(p.rows(),p.cols());
                  Matrix ddpRA = Matrix::Zero(p.rows(),p.cols());

                  Matrix d1k = Matrix::Zero(k.rows(),k.cols());
                  Matrix d2k = Matrix::Zero(k.rows(),k.cols());
                  Matrix ddk = Matrix::Zero(k.rows(),k.cols());
                  Matrix d1kQ = Matrix::Zero(k.rows(),k.cols());
                  Matrix d2kQ = Matrix::Zero(k.rows(),k.cols());
                  Matrix d1kR = Matrix::Zero(k.rows(),k.cols());
                  Matrix d2kR = Matrix::Zero(k.rows(),k.cols());
                  Matrix ddkQQ = Matrix::Zero(k.rows(),k.cols());
                  Matrix ddkQR = Matrix::Zero(k.rows(),k.cols());
                  Matrix ddkRQ = Matrix::Zero(k.rows(),k.cols());
                  Matrix ddkRR = Matrix::Zero(k.rows(),k.cols());
                  Matrix ddkAQ = Matrix::Zero(k.rows(),k.cols());
                  Matrix ddkQA = Matrix::Zero(k.rows(),k.cols());
                  Matrix ddkAR = Matrix::Zero(k.rows(),k.cols());
                  Matrix ddkRA = Matrix::Zero(k.rows(),k.cols());

                  for ( size_t t = 1; t < length; ++t ) {
                  //for ( size_t t = 1; t < 2; ++t ) {
                    //kalman
                    pt_1 = p;
                    //std::cout << "V(" << V_ir << "," << V_ic << ") k = " << k << std::endl;
                    p_1 = a*pt_1*a.transpose() + q;
                    kdenominator = (c*p_1*c.transpose()+r).inverse();
                    k = p_1*c.transpose()*kdenominator;
                    p = (id - k*c)*p_1;
                    x_1 = a*x.col(t-1) ;
                    x.col(t) = x_1 + k*(y.row(t) - (c*x_1).transpose()).transpose();
                    /*if( model.input ) {
                    hier muss noch rein, was mit input passiert...
                    }*/

                    //likelihood
                    likeE = y.row(t).transpose() - c * x_1; // - g * u.col(t);
                    likeS = c * p_1 * c.transpose() + r;
                    inverselikeS = likeS.inverse();

                    //first derivatives A
                    //kalman
                    eiejT = id.col(ir1)*id.row(ic1+floor(dimX/order)*iorder1);
                    d1pt_1 = d1p;
                    d1p_1 = eiejT*pt_1*a.transpose() + a*d1pt_1*a.transpose()+a*pt_1*eiejT.transpose();
                    d1k = (id - p_1*c.transpose()*kdenominator*c)*d1p_1*c.transpose()*kdenominator;
                    d1p = (id - k*c)*d1p_1 - d1k*c*p_1;
                    d1x_1 = eiejT*x.col(t-1) + a*d1x.col(t-1);
                    d1x.col(t) = (id - k*c)*d1x_1 + d1k*(y.row(t) - (c*x_1).transpose()).transpose();

                    //likelihood
                    d1likeE = -c * d1x_1;
                    d1likeS = c * d1p_1 * c.transpose();


                    if (iorder1 < 1 && iorder2 < 1) {

                      //first derivatives Q
                      //kalman
                      d1pt_1Q = d1pQ;
                      d1p_1Q = a*d1pt_1Q*a.transpose()+ eiejT;
                      d1kQ = (id - p_1*c.transpose()*kdenominator*c)*d1p_1Q*c.transpose()*kdenominator;
                      d1pQ = (id - k*c)*d1p_1Q - d1kQ*c*p_1;
                      d1x_1Q = a*d1xQ.col(t-1);
                      d1xQ.col(t) = (id - k*c)*d1x_1Q + d1kQ*(y.row(t) - (c*x_1).transpose()).transpose();

                      //likelihood
                      d1likeEQ = -c * d1x_1Q;
                      d1likeSQ = c * d1p_1Q * c.transpose();

                      //first derivatives R
                      //kalman
                      d1pt_1R = d1pR;
                      d1p_1R = a*d1pt_1R*a.transpose();
                      d1kR = (id - p_1*c.transpose()*kdenominator*c)*d1p_1R*c.transpose()*kdenominator - p_1*c.transpose()*kdenominator*eiejT.block(0,0,r.rows(),r.cols())*kdenominator;
                      d1pR = (id - k*c)*d1p_1R - d1kR*c*p_1;
                      d1x_1R = a*d1xR.col(t-1);
                      d1xR.col(t) = (id - k*c)*d1x_1R + d1kR*(y.row(t) - (c*x_1).transpose()).transpose();

                      //likelihood
                      d1likeER = -c * d1x_1R;
                      d1likeSR = c * d1p_1R * c.transpose() + eiejT.block(0,0,r.rows(),r.cols());


                     }


                    //second derivatives
                    //kalman
                    ekelT = id.col(ir2)*id.row(ic2+floor(dimX/order)*iorder2);
                    d2pt_1 = d2p;
                    d2p_1 = ekelT*pt_1*a.transpose() + a*d2pt_1*a.transpose()+a*pt_1*ekelT.transpose();
                    d2k = (id - p_1*c.transpose()*kdenominator*c)*d2p_1*c.transpose()*kdenominator;
                    d2p = (id - k*c)*d2p_1 - d2k*c*p_1;
                    d2x_1 = ekelT*x.col(t-1) + a*d2x.col(t-1);
                    d2x.col(t) = (id - k*c)*d2x_1 + d2k*(y.row(t) - (c*x_1).transpose()).transpose();

                    ddp_1 = eiejT*d2pt_1*a.transpose() + eiejT*pt_1*ekelT.transpose() + ekelT*d1pt_1*a.transpose() + a*ddp*a.transpose() + a*d1pt_1*ekelT.transpose() + ekelT*pt_1*eiejT.transpose() + a*d2pt_1*eiejT.transpose();
                    ddk = (ddp_1 - (id - p_1*c.transpose()*kdenominator*c)*d1p_1*c.transpose()*kdenominator*c*d2p_1 - (id - p_1*c.transpose()*kdenominator*c)*d2p_1*c.transpose()*kdenominator*c*d1p_1 - p_1*c.transpose()*kdenominator*c*ddp_1)*c.transpose()*kdenominator;
                    ddp = (id - k*c)*ddp_1 - d2k*c*d1p_1 - ddk*c*p_1 - d1k*c*d2p_1;
                    ddx_1 = ekelT*d1x.col(t-1) + a*ddx.col(t-1) + eiejT*d2x.col(t-1);
                    ddx.col(t) = (id - k*c)*ddx_1 - d2k*c*d1x_1 + ddk*(y.row(t) - (c*x_1).transpose()).transpose() - d1k*c*d2x_1;

                    //likelihood
                    d2likeE = -c * d2x_1;
                    d2likeS = c * d2p_1 * c.transpose();

                    ddlikeE = -c * ddx_1;
                    ddlikeS = c * ddp_1 * c.transpose();

                    logLiketmp1 = inverselikeS*ddlikeS - inverselikeS*d2likeS*inverselikeS*d1likeS;
                    logLiketmp2 = ddlikeE.transpose()*inverselikeS*likeE - d1likeE.transpose()*inverselikeS*d2likeS*inverselikeS*likeE + d1likeE.transpose()*inverselikeS*d2likeE - d2likeE.transpose()*inverselikeS*d1likeS*inverselikeS*likeE + likeE.transpose()*inverselikeS*d2likeS*inverselikeS*d1likeS*inverselikeS*likeE - likeE.transpose()*inverselikeS*ddlikeS*inverselikeS*likeE + likeE.transpose()*inverselikeS*d1likeS*inverselikeS*d2likeS*inverselikeS*likeE - likeE.transpose()*inverselikeS*d1likeS*inverselikeS*d2likeE + d2likeE.transpose()*inverselikeS*d1likeE - likeE.transpose()*inverselikeS*d2likeS*inverselikeS*d1likeE + likeE.transpose()*inverselikeS*ddlikeE;
                    logLike1 = logLiketmp1.trace();
                    logLike2 = logLiketmp2(0,0); //this line is necessary to make a number out of a 1x1 matrix

                    //std::cout << "V(" << V_ir << "," << V_ic << ")" << " ddp_1 = " << ddp_1 << std::endl;
                    V(V_ir,V_ic) += (logLike1 + logLike2)*(0.5/length);
                    V(V_ic,V_ir) = V(V_ir,V_ic);

                    //first derivatives Q II
                    //kalman
                    d2pt_1Q = d2pQ;
                    d2p_1Q = a*d2pt_1Q*a.transpose()+ ekelT;
                    d2kQ = (id - p_1*c.transpose()*kdenominator*c)*d2p_1Q*c.transpose()*kdenominator;
                    d2pQ = (id - k*c)*d2p_1Q - d2kQ*c*p_1;
                    d2x_1Q = a*d2xQ.col(t-1);
                    d2xQ.col(t) = (id - k*c)*d2x_1Q + d2kQ*(y.row(t) - (c*x_1).transpose()).transpose();

                    //likelihood
                    d2likeEQ = -c * d2x_1Q;
                    d2likeSQ = c * d2p_1Q * c.transpose();

                    //first derivatives R II
                    //kalman
                    d2pt_1R = d2pR;
                    d2p_1R = a*d2pt_1R*a.transpose();
                    d2kR = (id - p_1*c.transpose()*kdenominator*c)*d2p_1R*c.transpose()*kdenominator - p_1*c.transpose()*kdenominator*ekelT.block(0,0,r.rows(),r.cols())*kdenominator;
                    d2pR = (id - k*c)*d2p_1R - d2kR*c*p_1;
                    d2x_1R = a*d2xR.col(t-1);
                    d2xR.col(t) = (id - k*c)*d2x_1R + d2kR*(y.row(t) - (c*x_1).transpose()).transpose();

                    //likelihood
                    d2likeER = -c * d2x_1R;
                    d2likeSR = c * d2p_1R * c.transpose() + ekelT.block(0,0,r.rows(),r.cols());

                    if (iorder1 < 1 && iorder2 < 1) {

                      //second derivatives QQ
                      //kalman
                      ddp_1QQ = a*ddpQQ*a.transpose();
                      ddkQQ = (ddp_1QQ - (id - p_1*c.transpose()*kdenominator*c)*d1p_1Q*c.transpose()*kdenominator*c*d2p_1Q - (id - p_1*c.transpose()*kdenominator*c)*d2p_1Q*c.transpose()*kdenominator*c*d1p_1Q - p_1*c.transpose()*kdenominator*c*ddp_1QQ)*c.transpose()*kdenominator;
                      ddpQQ = (id - k*c)*ddp_1QQ - d2kQ*c*d1p_1Q - ddkQQ*c*p_1 - d1kQ*c*d2p_1Q;
                      ddx_1QQ = a*ddxQQ.col(t-1);
                      ddxQQ.col(t) = (id - k*c)*ddx_1QQ - d2kQ*c*d1x_1Q + ddkQQ*(y.row(t) - (c*x_1).transpose()).transpose() - d1kQ*c*d2x_1Q;

                      //likelihood
                      ddlikeEQQ = -c * ddx_1QQ;
                      ddlikeSQQ = c * ddp_1QQ * c.transpose();

                      logLiketmp1 = inverselikeS*ddlikeSQQ - inverselikeS*d2likeSQ*inverselikeS*d1likeSQ;
                      logLiketmp2 = ddlikeEQQ.transpose()*inverselikeS*likeE - d1likeEQ.transpose()*inverselikeS*d2likeSQ*inverselikeS*likeE + d1likeEQ.transpose()*inverselikeS*d2likeEQ - d2likeEQ.transpose()*inverselikeS*d1likeSQ*inverselikeS*likeE + likeE.transpose()*inverselikeS*d2likeSQ*inverselikeS*d1likeSQ*inverselikeS*likeE - likeE.transpose()*inverselikeS*ddlikeSQQ*inverselikeS*likeE + likeE.transpose()*inverselikeS*d1likeSQ*inverselikeS*d2likeSQ*inverselikeS*likeE - likeE.transpose()*inverselikeS*d1likeSQ*inverselikeS*d2likeEQ + d2likeEQ.transpose()*inverselikeS*d1likeEQ - likeE.transpose()*inverselikeS*d2likeSQ*inverselikeS*d1likeEQ + likeE.transpose()*inverselikeS*ddlikeEQQ;
                      logLike1 = logLiketmp1.trace();
                      logLike2 = logLiketmp2(0,0); //this line is necessary to make a number out of a 1x1 matrix

                      //std::cout << "QQ V(" << dimA+V_ir << "," << dimA+V_ic << ")" << std::endl;
                      V(dimA+V_ir,dimA+V_ic) += (logLike1 + logLike2)*(0.5/length);
                      V(dimA+V_ic,dimA+V_ir) = V(dimA+V_ir,dimA+V_ic);

                      //second derivatives QR
                      //kalman
                      ddp_1QR = a*ddpQR*a.transpose();
                      ddkQR = ddp_1QR*c.transpose()*kdenominator - d2p_1R*c.transpose()*kdenominator*c*d1p_1Q*c.transpose()*kdenominator - d1p_1Q*c.transpose()*kdenominator*(c*d2p_1R*c.transpose() + ekelT.block(0,0,r.rows(),r.cols()))*kdenominator + p_1*c.transpose()*kdenominator*c*d1p_1Q*c.transpose()*kdenominator*(c*d2p_1R*c.transpose() + ekelT.block(0,0,r.rows(),r.cols()))*kdenominator - p_1*c.transpose()*kdenominator*(c*ddp_1QR*c.transpose())*kdenominator + p_1*c.transpose()*kdenominator*(c*d2p_1R*c.transpose() + ekelT.block(0,0,r.rows(),r.cols()))*kdenominator*c*d1p_1Q*c.transpose()*kdenominator;
                      ddpQR = (id - k*c)*ddp_1QR - d2kR*c*d1p_1Q - ddkQR*c*p_1 - d1kQ*c*d2p_1R;
                      ddx_1QR = a*ddxQR.col(t-1);
                      ddxQR.col(t) = (id - k*c)*ddx_1QR - d2kR*c*d1x_1Q + ddkQR*(y.row(t) - (c*x_1).transpose()).transpose() - d1kQ*c*d2x_1R;

                      //likelihood
                      ddlikeEQR = -c * ddx_1QR;
                      ddlikeSQR = c * ddp_1QR * c.transpose();

                      logLiketmp1 = inverselikeS*ddlikeSQR - inverselikeS*d2likeSR*inverselikeS*d1likeSQ;
                      logLiketmp2 = ddlikeEQR.transpose()*inverselikeS*likeE - d1likeEQ.transpose()*inverselikeS*d2likeSR*inverselikeS*likeE + d1likeEQ.transpose()*inverselikeS*d2likeER - d2likeER.transpose()*inverselikeS*d1likeSQ*inverselikeS*likeE + likeE.transpose()*inverselikeS*d2likeSR*inverselikeS*d1likeSQ*inverselikeS*likeE - likeE.transpose()*inverselikeS*ddlikeSQR*inverselikeS*likeE + likeE.transpose()*inverselikeS*d1likeSQ*inverselikeS*d2likeSR*inverselikeS*likeE - likeE.transpose()*inverselikeS*d1likeSQ*inverselikeS*d2likeER + d2likeER.transpose()*inverselikeS*d1likeEQ - likeE.transpose()*inverselikeS*d2likeSR*inverselikeS*d1likeEQ + likeE.transpose()*inverselikeS*ddlikeEQR;
                      logLike1 = logLiketmp1.trace();
                      logLike2 = logLiketmp2(0,0); //this line is necessary to make a number out of a 1x1 matrix

                      //std::cout << "QR V(" << dimQ+dimA+V_ir << "," << dimA+V_ic << ")" << std::endl;
                      V(dimQ+dimA+V_ir,dimA+V_ic) += (logLike1 + logLike2)*(0.5/length);
                      V(dimA+V_ic,dimQ+dimA+V_ir) = V(dimQ+dimA+V_ir,dimA+V_ic);

                      if (V_ic < V_ir) {
                        //second derivatives RQ
                        //kalman
                        ddp_1RQ = a*ddpRQ*a.transpose();
                        ddkRQ = ddp_1RQ*c.transpose()*kdenominator - d1p_1R*c.transpose()*kdenominator*c*d2p_1Q*c.transpose()*kdenominator - d2p_1Q*c.transpose()*kdenominator*(c*d1p_1R*c.transpose() + eiejT.block(0,0,r.rows(),r.cols()))*kdenominator + p_1*c.transpose()*kdenominator*c*d2p_1Q*c.transpose()*kdenominator*(c*d1p_1R*c.transpose() + eiejT.block(0,0,r.rows(),r.cols()))*kdenominator - p_1*c.transpose()*kdenominator*(c*ddp_1RQ*c.transpose())*kdenominator + p_1*c.transpose()*kdenominator*(c*d1p_1R*c.transpose() + eiejT.block(0,0,r.rows(),r.cols()))*kdenominator*c*d2p_1Q*c.transpose()*kdenominator;
                        ddpRQ = (id - k*c)*ddp_1RQ - d1kR*c*d2p_1Q - ddkRQ*c*p_1 - d2kQ*c*d1p_1R;
                        ddx_1RQ = a*ddxRQ.col(t-1);
                        ddxRQ.col(t) = (id - k*c)*ddx_1RQ - d1kR*c*d2x_1Q + ddkRQ*(y.row(t) - (c*x_1).transpose()).transpose() - d2kQ*c*d1x_1R;

                        //likelihood
                        ddlikeERQ = -c * ddx_1RQ;
                        ddlikeSRQ = c * ddp_1RQ * c.transpose();

                        logLiketmp1 = inverselikeS*ddlikeSRQ - inverselikeS*d1likeSR*inverselikeS*d2likeSQ;
                        logLiketmp2 = ddlikeERQ.transpose()*inverselikeS*likeE - d2likeEQ.transpose()*inverselikeS*d1likeSR*inverselikeS*likeE + d2likeEQ.transpose()*inverselikeS*d1likeER - d1likeER.transpose()*inverselikeS*d2likeSQ*inverselikeS*likeE + likeE.transpose()*inverselikeS*d1likeSR*inverselikeS*d2likeSQ*inverselikeS*likeE - likeE.transpose()*inverselikeS*ddlikeSRQ*inverselikeS*likeE + likeE.transpose()*inverselikeS*d2likeSQ*inverselikeS*d1likeSR*inverselikeS*likeE - likeE.transpose()*inverselikeS*d2likeSQ*inverselikeS*d1likeER + d1likeER.transpose()*inverselikeS*d2likeEQ - likeE.transpose()*inverselikeS*d1likeSR*inverselikeS*d2likeEQ + likeE.transpose()*inverselikeS*ddlikeERQ;
                        logLike1 = logLiketmp1.trace();
                        logLike2 = logLiketmp2(0,0); //this line is necessary to make a number out of a 1x1 matrix

                        //std::cout << "RQ V(" << dimA+V_ir << "," << dimQ+dimA+V_ic << ")" << std::endl;
                        V(dimA+V_ir,dimQ+dimA+V_ic) += (logLike1 + logLike2)*(0.5/length);
                        V(dimQ+dimA+V_ic,dimA+V_ir) = V(dimA+V_ir,dimQ+dimA+V_ic);

                      }

                      //second derivatives RR
                      //kalman
                      ddp_1RR = a*ddpRR*a.transpose();
                      ddkRR = ddp_1RR*c.transpose()*kdenominator - d1p_1R*c.transpose()*kdenominator*(c*d2p_1R*c.transpose()+ekelT.block(0,0,r.rows(),r.cols()))*kdenominator - d2p_1R*c.transpose()*kdenominator*(c*d1p_1R*c.transpose() + eiejT.block(0,0,r.rows(),r.cols()))*kdenominator + p_1*c.transpose()*kdenominator*(c*d2p_1R*c.transpose() + ekelT.block(0,0,r.rows(),r.cols()))*kdenominator*(c*d1p_1R*c.transpose() + eiejT.block(0,0,r.rows(),r.cols()))*kdenominator - p_1*c.transpose()*kdenominator*(c*ddp_1RR*c.transpose())*kdenominator + p_1*c.transpose()*kdenominator*(c*d1p_1R*c.transpose() + eiejT.block(0,0,r.rows(),r.cols()))*kdenominator*(c*d2p_1R*c.transpose() + ekelT.block(0,0,r.rows(),r.cols()))*kdenominator;
                      ddpRR = (id - k*c)*ddp_1RR - d2kR*c*d1p_1R - ddkRR*c*p_1 - d1kR*c*d2p_1R;
                      ddx_1RR = a*ddxRR.col(t-1);
                      ddxRR.col(t) = (id - k*c)*ddx_1RR - d2kR*c*d1x_1R + ddkRR*(y.row(t) - (c*x_1).transpose()).transpose() - d1kR*c*d2x_1R;

                      //likelihood
                      ddlikeERR = -c * ddx_1RR;
                      ddlikeSRR = c * ddp_1RR * c.transpose();

                      logLiketmp1 = inverselikeS*ddlikeSRR - inverselikeS*d2likeSR*inverselikeS*d1likeSR;
                      logLiketmp2 = ddlikeERR.transpose()*inverselikeS*likeE - d1likeER.transpose()*inverselikeS*d2likeSR*inverselikeS*likeE + d1likeER.transpose()*inverselikeS*d2likeER - d2likeER.transpose()*inverselikeS*d1likeSR*inverselikeS*likeE + likeE.transpose()*inverselikeS*d2likeSR*inverselikeS*d1likeSR*inverselikeS*likeE - likeE.transpose()*inverselikeS*ddlikeSRR*inverselikeS*likeE + likeE.transpose()*inverselikeS*d1likeSR*inverselikeS*d2likeSR*inverselikeS*likeE - likeE.transpose()*inverselikeS*d1likeSR*inverselikeS*d2likeER + d2likeER.transpose()*inverselikeS*d1likeER - likeE.transpose()*inverselikeS*d2likeSR*inverselikeS*d1likeER + likeE.transpose()*inverselikeS*ddlikeERR;
                      logLike1 = logLiketmp1.trace();
                      logLike2 = logLiketmp2(0,0); //this line is necessary to make a number out of a 1x1 matrix

                      //std::cout << "RR V(" << dimQ+dimA+V_ir << "," << dimQ+dimA+V_ic << ")" << std::endl;
                      V(dimQ+dimA+V_ir,dimQ+dimA+V_ic) += (logLike1 + logLike2)*(0.5/length);
                      V(dimQ+dimA+V_ic,dimQ+dimA+V_ir) = V(dimQ+dimA+V_ir,dimQ+dimA+V_ic);


                    }

                    if (V_ic < dimQ) {
                      //second derivatives AQ
                      //kalman
                      ddp_1AQ = eiejT*d2p_1Q*a.transpose() + a*ddpAQ*a.transpose() + a*d2p_1Q*eiejT.transpose();
                      ddkAQ = (ddp_1AQ - (id - p_1*c.transpose()*kdenominator*c)*d1p_1*c.transpose()*kdenominator*c*d2p_1Q - (id - p_1*c.transpose()*kdenominator*c)*d2p_1Q*c.transpose()*kdenominator*c*d1p_1 - p_1*c.transpose()*kdenominator*c*ddp_1AQ)*c.transpose()*kdenominator;
                      ddpAQ = (id - k*c)*ddp_1AQ - d2kQ*c*d1p_1 - ddkAQ*c*p_1 - d1k*c*d2p_1Q;
                      ddx_1AQ = eiejT*d2x_1Q + a*ddxAQ.col(t-1);
                      ddxAQ.col(t) = (id - k*c)*ddx_1AQ - d2kQ*c*d1x_1 + ddkAQ*(y.row(t) - (c*x_1).transpose()).transpose() - d1k*c*d2x_1Q;

                      //likelihood
                      ddlikeEAQ = -c * ddx_1AQ;
                      ddlikeSAQ = c * ddp_1AQ * c.transpose();

                      logLiketmp1 = inverselikeS*ddlikeSAQ - inverselikeS*d2likeSQ*inverselikeS*d1likeS;
                      logLiketmp2 = ddlikeEAQ.transpose()*inverselikeS*likeE - d1likeE.transpose()*inverselikeS*d2likeSQ*inverselikeS*likeE + d1likeE.transpose()*inverselikeS*d2likeEQ - d2likeEQ.transpose()*inverselikeS*d1likeS*inverselikeS*likeE + likeE.transpose()*inverselikeS*d2likeSQ*inverselikeS*d1likeS*inverselikeS*likeE - likeE.transpose()*inverselikeS*ddlikeSAQ*inverselikeS*likeE + likeE.transpose()*inverselikeS*d1likeS*inverselikeS*d2likeSQ*inverselikeS*likeE - likeE.transpose()*inverselikeS*d1likeS*inverselikeS*d2likeEQ + d2likeEQ.transpose()*inverselikeS*d1likeE - likeE.transpose()*inverselikeS*d2likeSQ*inverselikeS*d1likeE + likeE.transpose()*inverselikeS*ddlikeEAQ;
                      logLike1 = logLiketmp1.trace();
                      logLike2 = logLiketmp2(0,0); //this line is necessary to make a number out of a 1x1 matrix

                      //std::cout << "AQ V(" << V_ir << "," << dimA+V_ic << ")" << std::endl;
                      V(V_ir,dimA+V_ic) += (logLike1 + logLike2)*(0.5/length);
                      V(dimA+V_ic,V_ir) = V(V_ir,dimA+V_ic);

                      if (iorder1 < 1 && V_ic < V_ir) {
                        //second derivatives QA
                        //kalman
                        ddp_1QA = ekelT*d1p_1Q*a.transpose() + a*ddpQA*a.transpose() + a*d1p_1Q*ekelT.transpose();
                        ddkQA = (ddp_1QA - (id - p_1*c.transpose()*kdenominator*c)*d2p_1*c.transpose()*kdenominator*c*d1p_1Q - (id - p_1*c.transpose()*kdenominator*c)*d1p_1Q*c.transpose()*kdenominator*c*d2p_1 - p_1*c.transpose()*kdenominator*c*ddp_1QA)*c.transpose()*kdenominator;
                        ddpQA = (id - k*c)*ddp_1QA - d1kQ*c*d2p_1 - ddkQA*c*p_1 - d2k*c*d1p_1Q;
                        ddx_1QA = ekelT*d1x_1Q + a*ddxQA.col(t-1);
                        ddxQA.col(t) = (id - k*c)*ddx_1QA - d1kQ*c*d2x_1 + ddkQA*(y.row(t) - (c*x_1).transpose()).transpose() - d2k*c*d1x_1Q;

                        //likelihood
                        ddlikeEQA = -c * ddx_1QA;
                        ddlikeSQA = c * ddp_1QA * c.transpose();

                        logLiketmp1 = inverselikeS*ddlikeSQA - inverselikeS*d1likeSQ*inverselikeS*d2likeS;
                        logLiketmp2 = ddlikeEQA.transpose()*inverselikeS*likeE - d2likeE.transpose()*inverselikeS*d1likeSQ*inverselikeS*likeE + d2likeE.transpose()*inverselikeS*d1likeEQ - d1likeEQ.transpose()*inverselikeS*d2likeS*inverselikeS*likeE + likeE.transpose()*inverselikeS*d1likeSQ*inverselikeS*d2likeS*inverselikeS*likeE - likeE.transpose()*inverselikeS*ddlikeSQA*inverselikeS*likeE + likeE.transpose()*inverselikeS*d2likeS*inverselikeS*d1likeSQ*inverselikeS*likeE - likeE.transpose()*inverselikeS*d2likeS*inverselikeS*d1likeEQ + d1likeEQ.transpose()*inverselikeS*d2likeE - likeE.transpose()*inverselikeS*d1likeSQ*inverselikeS*d2likeE + likeE.transpose()*inverselikeS*ddlikeEQA;
                        logLike1 = logLiketmp1.trace();
                        logLike2 = logLiketmp2(0,0); //this line is necessary to make a number out of a 1x1 matrix

                        //std::cout << "QA V(" << dimA+V_ir << "," << V_ic << ")" << std::endl;
                        V(dimA+V_ir,V_ic) += (logLike1 + logLike2)*(0.5/length);
                        V(V_ic,dimA+V_ir) = V(dimA+V_ir,V_ic);

                      }

                      //second derivatives AR
                      //kalman
                      ddp_1AR = eiejT*d2p_1R*a.transpose() + a*ddpAR*a.transpose() + a*d2p_1R*eiejT.transpose();
                      ddkAR = ddp_1AR*c.transpose()*kdenominator - d2p_1R*c.transpose()*kdenominator*c*d1p_1*c.transpose()*kdenominator - d1p_1*c.transpose()*kdenominator*(c*d2p_1R*c.transpose() + ekelT.block(0,0,r.rows(),r.cols()))*kdenominator + p_1*c.transpose()*kdenominator*c*d1p_1*c.transpose()*kdenominator*(c*d2p_1R*c.transpose() + ekelT.block(0,0,r.rows(),r.cols()))*kdenominator - p_1*c.transpose()*kdenominator*(c*ddp_1AR*c.transpose())*kdenominator + p_1*c.transpose()*kdenominator*(c*d2p_1R*c.transpose() + ekelT.block(0,0,r.rows(),r.cols()))*kdenominator*c*d1p_1*c.transpose()*kdenominator;
                      ddpAR = (id - k*c)*ddp_1AR - d2kR*c*d1p_1 - ddkAR*c*p_1 - d1k*c*d2p_1R;
                      ddx_1AR = eiejT*d2x_1R + a*ddxAR.col(t-1);
                      ddxAR.col(t) = (id - k*c)*ddx_1AR - d2kR*c*d1x_1 + ddkAR*(y.row(t) - (c*x_1).transpose()).transpose() - d1k*c*d2x_1R;

                      //likelihood
                      ddlikeEAR = -c * ddx_1AR;
                      ddlikeSAR = c * ddp_1AR * c.transpose();

                      logLiketmp1 = inverselikeS*ddlikeSAR - inverselikeS*d2likeSR*inverselikeS*d1likeS;
                      logLiketmp2 = ddlikeEAR.transpose()*inverselikeS*likeE - d1likeE.transpose()*inverselikeS*d2likeSR*inverselikeS*likeE + d1likeE.transpose()*inverselikeS*d2likeER - d2likeER.transpose()*inverselikeS*d1likeS*inverselikeS*likeE + likeE.transpose()*inverselikeS*d2likeSR*inverselikeS*d1likeS*inverselikeS*likeE - likeE.transpose()*inverselikeS*ddlikeSAR*inverselikeS*likeE + likeE.transpose()*inverselikeS*d1likeS*inverselikeS*d2likeSR*inverselikeS*likeE - likeE.transpose()*inverselikeS*d1likeS*inverselikeS*d2likeER + d2likeER.transpose()*inverselikeS*d1likeE - likeE.transpose()*inverselikeS*d2likeSR*inverselikeS*d1likeE + likeE.transpose()*inverselikeS*ddlikeEAR;
                      logLike1 = logLiketmp1.trace();
                      logLike2 = logLiketmp2(0,0); //this line is necessary to make a number out of a 1x1 matrix

                      //std::cout << "AR V(" << V_ir << "," << dimQ+dimA+V_ic << ")" << std::endl;
                      V(V_ir,dimQ+dimA+V_ic) += (logLike1 + logLike2)*(0.5/length);
                      V(dimQ+dimA+V_ic,V_ir) = V(V_ir,dimQ+dimA+V_ic);

                      if (iorder1 < 1 && V_ic < V_ir) {
                        //second derivatives RA
                        //kalman
                        ddp_1RA = ekelT*d1p_1R*a.transpose() + a*ddpRA*a.transpose() + a*d1p_1R*ekelT.transpose();
                        ddkRA = ddp_1RA*c.transpose()*kdenominator - d1p_1R*c.transpose()*kdenominator*c*d2p_1*c.transpose()*kdenominator - d2p_1*c.transpose()*kdenominator*(c*d1p_1R*c.transpose() + eiejT.block(0,0,r.rows(),r.cols()))*kdenominator + p_1*c.transpose()*kdenominator*c*d2p_1*c.transpose()*kdenominator*(c*d1p_1R*c.transpose() + eiejT.block(0,0,r.rows(),r.cols()))*kdenominator - p_1*c.transpose()*kdenominator*(c*ddp_1RA*c.transpose())*kdenominator + p_1*c.transpose()*kdenominator*(c*d1p_1R*c.transpose() + eiejT.block(0,0,r.rows(),r.cols()))*kdenominator*c*d2p_1*c.transpose()*kdenominator;
                        ddpRA = (id - k*c)*ddp_1RA - d1kR*c*d2p_1 - ddkRA*c*p_1 - d2k*c*d1p_1R;
                        ddx_1RA = ekelT*d1x_1R + a*ddxRA.col(t-1);
                        ddxRA.col(t) = (id - k*c)*ddx_1RA - d1kR*c*d2x_1 + ddkRA*(y.row(t) - (c*x_1).transpose()).transpose() - d2k*c*d1x_1R;

                        //likelihood
                        ddlikeERA = -c * ddx_1RA;
                        ddlikeSRA = c * ddp_1RA * c.transpose();

                        logLiketmp1 = inverselikeS*ddlikeSRA - inverselikeS*d1likeSR*inverselikeS*d2likeS;
                        logLiketmp2 = ddlikeERA.transpose()*inverselikeS*likeE - d2likeE.transpose()*inverselikeS*d1likeSR*inverselikeS*likeE + d2likeE.transpose()*inverselikeS*d1likeER - d1likeER.transpose()*inverselikeS*d2likeS*inverselikeS*likeE + likeE.transpose()*inverselikeS*d1likeSR*inverselikeS*d2likeS*inverselikeS*likeE - likeE.transpose()*inverselikeS*ddlikeSRA*inverselikeS*likeE + likeE.transpose()*inverselikeS*d2likeS*inverselikeS*d1likeSR*inverselikeS*likeE - likeE.transpose()*inverselikeS*d2likeS*inverselikeS*d1likeER + d1likeER.transpose()*inverselikeS*d2likeE - likeE.transpose()*inverselikeS*d1likeSR*inverselikeS*d2likeE + likeE.transpose()*inverselikeS*ddlikeERA;
                        logLike1 = logLiketmp1.trace();
                        logLike2 = logLiketmp2(0,0); //this line is necessary to make a number out of a 1x1 matrix

                        //std::cout << "RA V(" << dimQ+dimA+V_ir << "," << V_ic << ")" << std::endl;
                        V(dimQ+dimA+V_ir,V_ic) += (logLike1 + logLike2)*(0.5/length);
                        V(V_ic,dimQ+dimA+V_ir) = V(dimQ+dimA+V_ir,V_ic);


                      }
                    }

                  }
                }
              }
            }
          }
        }
      }

    }

    }

    Matrix V;

protected:
    std::vector< Panel > * panels;
    Model * model;
};


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


static void logV_analytically( std::vector< Panel > & panels, Model & model, Info & info ) {
    std::clock_t now = std::clock();

    std::ofstream logA( info.logFiles["A"].c_str(), std::ofstream::app ),
        logQ( info.logFiles["Q"].c_str(), std::ofstream::app ),
        logR( info.logFiles["R"].c_str(), std::ofstream::app ),
  logV( info.logFiles["V"].c_str(), std::ofstream::app );

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
    info.checkpoint = now;
}

void V_analytically( std::vector< Panel > & panels, Model & model, Info & info ) {

  calc_dd_incompl_likelihood ddlnlike( panels, model );
  tbb::parallel_reduce( tbb::blocked_range< size_t >( 0, panels.size() ), ddlnlike );

  Matrix inverted_ddlikeV;
  inverted_ddlikeV = (ddlnlike.V).inverse();
  model.v = inverted_ddlikeV.diagonal();

  logV_analytically( panels, model, info );
}
