//
// Created by arqwer on 18.04.17.
//

#ifndef MY_SLAM_JACOBIANS_H
#define MY_SLAM_JACOBIANS_H

#include "my_types.h"

#include <stdint.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//13 x 6
Mat Q_df_over_dn(const StateMean &state, double dt);


//13 x 13
Mat Ft_df_over_dxcam(const StateMean &state, double dt);

//200 x 313
//2N x 3N+13
Mat H_t_Jacobian_of_observations(const StateMean &s, const Mat &camIntrinsics);


#endif //MY_SLAM_JACOBIANS_H
