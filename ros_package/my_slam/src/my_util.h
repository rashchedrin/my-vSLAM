//
// Created by arqwer on 18.04.17.
//

#ifndef MY_SLAM_MY_UTIL_H
#define MY_SLAM_MY_UTIL_H

#include <stdint.h>
#include "opencv2/opencv.hpp"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

using namespace cv;
using namespace std;

const long double pi = 3.141592653589793238462643383279502884L /* pi */;

uint32_t uinthash(uint32_t x);

int32_t inthash(int32_t val, int32_t salt = 98262, int32_t low = 0, int32_t high = 256);

Scalar hashcolor(int32_t val, int32_t salt = 0);

void display_mat(Mat m, string name = "", bool cformat = false);

void DrawCross(Mat output_mat, Point2d pt, Scalar color = Scalar(0, 255, 0), int size = 5);

void DrawPoints(Mat &output_image,
                const vector<Point2d> &points_coords,
                char marker_type = 'o');

Mat ImageFromMsg(const sensor_msgs::ImageConstPtr &msg);

vector<Point2d> GetMatchingPointsCoordinates(const vector<KeyPoint> &key_points,
                                             const Mat &kp_descriptors,
                                             const Mat &known_descriptors,
                                             const NormTypes &norm_type = NORM_HAMMING);

double limitPi(double a);
#endif //MY_SLAM_MY_UTIL_H
