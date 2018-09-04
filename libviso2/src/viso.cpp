/*
Copyright 2011. All rights reserved.
Institute of Measurement and Control Systems
Karlsruhe Institute of Technology, Germany

This file is part of libviso2.
Authors: Andreas Geiger

libviso2 is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

libviso2 is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
libviso2; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA 
*/

#include "viso.h"
#include "iostream"
#include <math.h>

using namespace std;

VisualOdometry::VisualOdometry(parameters param) : _param(param)
{
    _J         = 0;
    _p_observe = 0;
    _p_predict = 0;
    _matcher   = new Matcher(param.match);
    _Tr_delta  = libviso2_Matrix::eye(4);
    _Tr_valid  = false;
    srand(0);
}

//==============================================================================//

VisualOdometry::~VisualOdometry()
{
    delete _matcher;
}

//==============================================================================//

bool VisualOdometry::updateMotion()
{
    // estimate motion
    vector<double> tr_delta = estimateMotion(_p_matched);

    // on failure
    if (tr_delta.size()!=6)
    {
        return false;
    }

    // set transformation libviso2_Matrix (previous to current frame)
    _Tr_delta = transformationVectorToMatrix(tr_delta);
    _Tr_valid = true;

    // success
    return true;
}

//==============================================================================//

libviso2_Matrix VisualOdometry::transformationVectorToMatrix(vector<double> tr)
{
    // extract parameters
    double rx = tr[0];
    double ry = tr[1];
    double rz = tr[2];
    double tx = tr[3];
    double ty = tr[4];
    double tz = tr[5];


    // precompute sine/cosine
    double sx = sin(rx);
    double cx = cos(rx);
    double sy = sin(ry);
    double cy = cos(ry);
    double sz = sin(rz);
    double cz = cos(rz);


    // compute transformation
    libviso2_Matrix Tr(4,4);
    Tr._val[0][0] = +cy*cz;          Tr._val[0][1] = -cy*sz;          Tr._val[0][2] = +sy;    Tr._val[0][3] = tx;
    Tr._val[1][0] = +sx*sy*cz+cx*sz; Tr._val[1][1] = -sx*sy*sz+cx*cz; Tr._val[1][2] = -sx*cy; Tr._val[1][3] = ty;
    Tr._val[2][0] = -cx*sy*cz+sx*sz; Tr._val[2][1] = +cx*sy*sz+sx*cz; Tr._val[2][2] = +cx*cy; Tr._val[2][3] = tz;
    Tr._val[3][0] = 0;               Tr._val[3][1] = 0;               Tr._val[3][2] = 0;      Tr._val[3][3] = 1;

   /* libviso2_Matrix BR(1,12);
    BR._val[0][0] = +cy*cz;          BR._val[0][1] = -cy*sz;          BR._val[0][2] = +sy;    BR._val[0][3] = tx;
    BR._val[0][4] = +sx*sy*cz+cx*sz; BR._val[0][5] = -sx*sy*sz+cx*cz; BR._val[0][6] = -sx*cy; BR._val[0][7] = ty;
    BR._val[0][8] = -cx*sy*cz+sx*sz; BR._val[0][9] = +cx*sy*sz+sx*cz; BR._val[0][10] = +cx*cy; BR._val[0][11] = tz;
    std::cout << Tr << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;*/

    return Tr;
}

//==============================================================================//

vector<int32_t> VisualOdometry::getRandomSample(int32_t N,int32_t num)
{
    // init sample and totalset
    vector<int32_t> sample;
    vector<int32_t> totalset;

    // create vector containing all indices
    for (int32_t i=0; i<N; i++)
    totalset.push_back(i);

    // add num indices to current sample
    sample.clear();
    for (int32_t i=0; i<num; i++)
    {
        int32_t j = rand()%totalset.size();
        sample.push_back(totalset[j]);
        totalset.erase(totalset.begin()+j);
    }

    // return sample
    return sample;
}
