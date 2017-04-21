//
// Created by arqwer on 18.04.17.
//

#ifndef MY_SLAM_MY_TYPES_H
#define MY_SLAM_MY_TYPES_H

#include <ostream>
#include "opencv2/opencv.hpp"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;
using namespace cv;

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

class Quaternion {
 public:
  double data[4];

  double &w() {
    return data[0];
  }

  double &i() {
    return data[1];
  }

  double &j() {
    return data[2];
  }

  double &k() {
    return data[3];
  }

  const double w() const {
    return data[0];
  }

  const double i() const {
    return data[1];
  }

  const double j() const {
    return data[2];
  }

  const double k() const {
    return data[3];
  }

  Quaternion() {}

  //Get Quaternion from angles
  explicit Quaternion(Point3d axis_angle) {
    double theta = norm(axis_angle);
    Point3d unit_vec = axis_angle / theta;
    w() = cos(theta / 2);
    i() = unit_vec.x * sin(theta / 2);
    j() = unit_vec.y * sin(theta / 2);
    k() = unit_vec.z * sin(theta / 2);
  }

  Quaternion(double w, double i, double j, double k) {
    data[0] = w;
    data[1] = i;
    data[2] = j;
    data[3] = k;
  }

  double &operator[](int k) {
    return data[k];
  }

  double operator[](int k) const {
    return data[k];
  }

  friend ostream &operator<<(ostream &os, const Quaternion &quaternion) {
    os << quaternion.w() << " + " << quaternion.i() << "i + " << quaternion.j() << "j + "
       << quaternion.k() << "k";
    return os;
  }
};

Quaternion operator+(const Quaternion &a, const Quaternion &b);
Quaternion operator*(const Quaternion &q1, const Quaternion &q2);

Quaternion operator*(const Quaternion &q1, double val);
Quaternion operator*(double val, const Quaternion &q);
Quaternion operator/(const Quaternion &q, double val);

double norm(const Quaternion &q);

struct StateMean {
  // w = world
  // r = relative
  Point3d position_w;         // 3
  Quaternion direction_w;     // 4
  Point3d velocity_w;         // 3
  Point3d angular_velocity_r; // 3
  std::vector<Point3d> feature_positions_w;  // 3*n
  //total = 13+3*n
  // /.n->4 => total = 25
  StateMean() {}
  explicit StateMean(Mat m);
  friend ostream &operator<<(ostream &os, const StateMean &stateMean);
};

Mat state2mat(StateMean s);

void StateToMsg(const StateMean &s, vector<Point3d> trajectory, PointCloud *points3D);

#endif //MY_SLAM_MY_TYPES_H
