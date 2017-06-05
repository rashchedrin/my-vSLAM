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
#include "my_types.h"

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
                char marker_type, int size, const vector<PointStatistics> &pt_stats);

Mat ImageFromMsg(const sensor_msgs::ImageConstPtr &msg);

vector<Point2d> GetMatchingPointsCoordinates(const vector<KeyPoint> &key_points,
                                             const Mat &kp_descriptors,
                                             const Mat &known_descriptors,
                                             const vector<Point2d> &expected_positions,
                                             int search_radius,
                                             const NormTypes &norm_type,
                                             vector<bool> *isFound,
                                             vector<double> *descr_unsimilarity);

double limitPi(double a);

double NormalPdf2d(const Mat &sigma, Vec2d mean, Vec2d x);

Point2i L2MatchingPosition(const Mat &image, const Mat &patch, Rect search_region);

//double norm(const vector<double> &vec);

//double sum(const vector<double> &vec);

void normalize(vector<double> *vec);

void normalize(vector<double> *vec, double divisor);

Mat L2DistanceMat(const Mat &image, const Mat &patch);
Point2i ArgMin(Mat values);
Mat CovarianceAlongLine(double x, double y, double z, double dispersion, double perp_dispersion);

vector<Point2d> ToSparseVec(const vector<Point2d> &full_vec, const vector<bool> &is_included);

bool isTriangular(const Mat &m, double eps = 0.00001);

bool Triangulize(Mat *m, double eps = 0.00001);

bool isSemiPositive(const Mat &m);

#endif //MY_SLAM_MY_UTIL_H
