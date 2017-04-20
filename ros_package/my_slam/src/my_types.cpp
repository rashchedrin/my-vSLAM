//
// Created by arqwer on 18.04.17.
//

#include "my_types.h"
#include "my_util.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Quaternion operator+(const Quaternion &a, const Quaternion &b) {
  Quaternion res;
  for (int i = 0; i < 4; i++) {
    res[i] = a[i] + b[i];
  }
  return res;
}

//source: http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/code/
Quaternion operator*(const Quaternion &q1, const Quaternion &q2) {
  Quaternion res;
  res[1] = q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2] + q1[0] * q2[1];
  res[2] = -q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1] + q1[0] * q2[2];
  res[3] = q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0] + q1[0] * q2[3];
  res[0] = -q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3] + q1[0] * q2[0];
  return res;
}

Quaternion operator*(const Quaternion &q, double val){
  return Quaternion(q.w() * val, q.i() * val, q.j() * val, q.k() * val);
}
Quaternion operator*(double val, const Quaternion &q){
  return q * val;
}
Quaternion operator/(const Quaternion &q, double val){
  return q * (1.0/val);
}

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

StateMean::StateMean(Mat m) {
  assert(m.rows >= 13);
  int n_points = (m.rows - 13) / 3;
  double rx = m.at<double>(0, 0);
  double ry = m.at<double>(1, 0);
  double rz = m.at<double>(2, 0);
  double qw = m.at<double>(3, 0);
  double qi = m.at<double>(4, 0);
  double qj = m.at<double>(5, 0);
  double qk = m.at<double>(6, 0);
  double vx = m.at<double>(7, 0);
  double vy = m.at<double>(8, 0);
  double vz = m.at<double>(9, 0);
  double wx = m.at<double>(10, 0);
  double wy = m.at<double>(11, 0);
  double wz = m.at<double>(12, 0);
  position_w = Point3d(rx, ry, rz);
  direction_w = Quaternion(qw, qi, qj, qk);
  velocity_w = Point3d(vx, vy, vz);
  angular_velocity_r = Point3d(wx, wy, wz);
  for (int i_pt = 0; i_pt < n_points; ++i_pt) {
    double px = m.at<double>(13 + i_pt * 3 + 0, 0);
    double py = m.at<double>(13 + i_pt * 3 + 1, 0);
    double pz = m.at<double>(13 + i_pt * 3 + 2, 0);
    feature_positions_w.push_back(Point3d(px, py, pz));
  }
}
ostream &operator<<(ostream &os, const StateMean &stateMean) {
  os << "position_w:\t" << stateMean.position_w <<endl
     << "direction_w:\t" << stateMean.direction_w <<" phi = "<<limitPi(acos(stateMean.direction_w.w())*2)<<endl
     << "velocity_w:\t" << stateMean.velocity_w <<endl
     << "angular_velocity_r:\t"  << stateMean.angular_velocity_r<<" phi = "<<limitPi(norm(stateMean.angular_velocity_r) )<<endl
     << "feature_positions_w:\n" << stateMean.feature_positions_w;
  return os;
}

double norm(const Quaternion &q){
  return sqrt(q.w() * q.w() + q.i() * q.i() + q.j() * q.j() + q.k() * q.k());
}