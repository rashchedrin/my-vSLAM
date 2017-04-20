//
// Created by arqwer on 18.04.17.
//

#include "my_util.h"

#include <stdint.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

using namespace cv;
using namespace std;

uint32_t uinthash(uint32_t x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}

int32_t inthash(int32_t val, int32_t salt, int32_t low, int32_t high) {
  val = uinthash(val) + uinthash(uinthash(salt));
  val = val < 0 ? -val : val;
  return (val % (high - low)) + low;
}

Scalar hashcolor(int32_t val, int32_t salt) {
  if(salt == 0 ) {
    switch (val) {
      case 0: return Scalar(256, 0, 0);
        break;
      case 1: return Scalar(0, 256, 0);
        break;
      case 2: return Scalar(0, 0, 256);
        break;
      case 3: return Scalar(0, 0, 0);
        break;
    }
  }
  return Scalar({inthash(val, inthash(salt + 1)), inthash(val, inthash(salt + 2)),
                 inthash(val, inthash(salt + 3))});
}

void display_mat(Mat m, string name, bool cformat) {
  cout << name << " " << m.rows << " x " << m.cols << endl;
  if(cformat){
    cout<<"{"<<endl;
  }
  for (int ir = 0; ir < m.rows; ++ir) {
    if(cformat){
      cout<<"{ ";
    }
    for (int ic = 0; ic < m.cols; ++ic) {
      cout << m.at<double>(ir, ic);
      if(cformat && ic != m.cols - 1){
        cout<<",";
      }
      cout<< " ";
    }
    if(cformat){
      cout<<"}";
      if(ir != m.rows - 1){
        cout<<",";
      }
    }
    cout << endl;
  }
  if(cformat){
    cout<<"}"<<endl;
  }
}


void DrawCross(Mat output_mat, Point2d pt, Scalar color, int size){
  Point2d s1 = Point2d(size, size);
  Point2d s2 = Point2d(size, -size);
  line(output_mat, pt - s1, pt + s1, color, 2);
  line(output_mat, pt - s2, pt + s2, color, 2);
  line(output_mat, pt - s1, pt + s1, Scalar(255,255,255), 1);
  line(output_mat, pt - s2, pt + s2, Scalar(255,255,255), 1);
}


void DrawPoints(Mat &output_image,
                const vector<Point2d> &points_coords,
                char marker_type) {
  for (int i_obs = 0; i_obs < points_coords.size(); ++i_obs) {
    Scalar color = hashcolor(i_obs);
    if (marker_type == 'o') {
      circle(output_image, points_coords[i_obs], 4, color, 7);
    }
    if (marker_type == 'x') {
      DrawCross(output_image, points_coords[i_obs], color);
    }
  }
}

Mat ImageFromMsg(const sensor_msgs::ImageConstPtr &msg){
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    exit(1);
  }
  return cv_ptr->image;
}

vector<Point2d> GetMatchingPointsCoordinates(const vector<KeyPoint> &key_points,
                                             const Mat &kp_descriptors,
                                             const Mat &known_descriptors,
                                             const NormTypes &norm_type) {
  vector<Point2d> coordinates_vec;
  for (int i_known_kp = 0; i_known_kp < 4; ++i_known_kp) {
    double min_distance =
        norm(known_descriptors.row(i_known_kp), kp_descriptors.row(0), norm_type);
    int closest_id = 0;
    for (int i_kp = 0; i_kp < key_points.size(); ++i_kp) {
      double distance =
          norm(known_descriptors.row(i_known_kp), kp_descriptors.row(i_kp), norm_type);
      if (distance < min_distance) {
        min_distance = distance;
        closest_id = i_kp;
      }
    }
    coordinates_vec.push_back(key_points[closest_id].pt);
  }
  return coordinates_vec;
}

double mod(double a, double b){
  while(a > b){
    a -= b;
  }
  return a;
}

double limitPi(double a){
  a = mod(a, pi * 2);
  if(a > pi){
    a -= 2*pi;
  }
  return a;
}