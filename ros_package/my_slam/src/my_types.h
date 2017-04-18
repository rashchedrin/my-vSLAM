//
// Created by arqwer on 18.04.17.
//

#ifndef MY_SLAM_MY_TYPES_H
#define MY_SLAM_MY_TYPES_H

#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;



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
  Quaternion(Point3d axis_angle) {
    double theta = norm(axis_angle);
    Point3d unit_vec = axis_angle / theta;
    w() = cos(theta / 2);
    i() = unit_vec.x * sin(theta / 2);
    j() = unit_vec.y * sin(theta / 2);
    k() = unit_vec.z * sin(theta / 2);
  }

  Quaternion(int w, int i, int j, int k) {
    data[0] = w;
    data[1] = i;
    data[2] = j;
    data[3] = k;
  }

  double &operator[](int k) {
    return data[k];
  }

};

Quaternion operator+(Quaternion a, Quaternion b);
Quaternion operator*(Quaternion q1, Quaternion q2);

Quaternion operator+(Quaternion a, Quaternion b) {
  Quaternion res;
  for (int i = 0; i < 4; i++) {
    res[i] = a[i] + b[i];
  }
  return res;
}

//source: http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/code/
Quaternion operator*(Quaternion q1, Quaternion q2) {
  Quaternion res;
  res[1] = q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2] + q1[0] * q2[1];
  res[2] = -q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1] + q1[0] * q2[2];
  res[3] = q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0] + q1[0] * q2[3];
  res[0] = -q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3] + q1[0] * q2[0];
  return res;
}

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
  StateMean(){}
  StateMean(Mat m){
    assert(m.rows >= 13);
    int n_points = (m.rows -13)/3;
    double rx = m.at<double>(0,0);
    double ry = m.at<double>(1,0);
    double rz = m.at<double>(2,0);
    double qw = m.at<double>(3,0);
    double qi = m.at<double>(4,0);
    double qj = m.at<double>(5,0);
    double qk = m.at<double>(6,0);
    double vx = m.at<double>(7,0);
    double vy = m.at<double>(8,0);
    double vz = m.at<double>(9,0);
    double wx = m.at<double>(10,0);
    double wy = m.at<double>(11,0);
    double wz = m.at<double>(12,0);
    position_w = Point3d(rx,ry,rz);
    direction_w = Quaternion(qw,qi,qj,qk);
    velocity_w = Point3d(vx,vy,vz);
    angular_velocity_r = Point3d(wx,wy,wz);
    for(int i_pt = 0; i_pt < n_points; ++i_pt){
      double px = m.at<double>(13 + i_pt * 3 + 0,0);
      double py = m.at<double>(13 + i_pt * 3 + 1,0);
      double pz = m.at<double>(13 + i_pt * 3 + 2,0);
      feature_positions_w.push_back(Point3d(px,py,pz));
    }
  }
};

Mat state2mat(StateMean s){
  int size = 13 + 3 * s.feature_positions_w.size();
  Mat result(size,1,CV_64F);
  result.at<double>(0,0) = s.position_w.x;
  result.at<double>(1,0) = s.position_w.y;
  result.at<double>(2,0) = s.position_w.z;

  result.at<double>(3,0) = s.direction_w.w();
  result.at<double>(4,0) = s.direction_w.i();
  result.at<double>(5,0) = s.direction_w.j();
  result.at<double>(6,0) = s.direction_w.k();

  result.at<double>(7,0) = s.velocity_w.x;
  result.at<double>(8,0) = s.velocity_w.y;
  result.at<double>(9,0) = s.velocity_w.z;
  result.at<double>(10,0) = s.angular_velocity_r.x;
  result.at<double>(11,0) = s.angular_velocity_r.y;
  result.at<double>(12,0) = s.angular_velocity_r.z;
  for (int i_pt = 0; i_pt < s.feature_positions_w.size(); ++i_pt){
    result.at<double>(13 + i_pt * 3 + 0, 0) = s.feature_positions_w[i_pt].x;
    result.at<double>(13 + i_pt * 3 + 1, 0) = s.feature_positions_w[i_pt].y;
    result.at<double>(13 + i_pt * 3 + 2, 0) = s.feature_positions_w[i_pt].z;
  }
  return result;
}

#endif //MY_SLAM_MY_TYPES_H
