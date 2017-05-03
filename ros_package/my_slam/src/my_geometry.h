//
// Created by arqwer on 23.04.17.
//

#ifndef MY_SLAM_MY_GEOMETRY_H
#define MY_SLAM_MY_GEOMETRY_H

#include "my_types.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Vec3d RayFromXY_rel(int x, int y, const Mat &camIntrinsics);

Point2d ProjectPoint(Point3d pt_3d,
                     Point3d cam_position,
                     const Quaternion &rotation_wr,
                     const Mat &camIntrinsics);

Mat RotationMatXtoXYZ(double x, double y, double z);

#endif //MY_SLAM_MY_GEOMETRY_H
