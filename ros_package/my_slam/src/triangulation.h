//
// Created by arqwer on 05.06.17.
//

#ifndef MY_SLAM_TRIANGULATION_H
#define MY_SLAM_TRIANGULATION_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;
using namespace cv;

void Triangulate2Frames(cv::Mat frame1, cv::Mat frame2, const Mat & camera_intrinsic,vector<pcl::PointXYZRGB> *points, Mat out_points_cov);

#endif //MY_SLAM_TRIANGULATION_H
