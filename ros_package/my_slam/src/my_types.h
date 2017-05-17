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
const int N_VARS_FOR_CAMERA = 13;
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

//Get Quaternion from angles
Quaternion QuaternionFromAxisAngle(Point3d axis_angle);

Quaternion conj(const Quaternion &q);

Quaternion Vec2Quat(const Vec3d &v);

Vec3d RotateByQuat(const Quaternion &q, const Vec3d &v);
Vec3d Direction(const Quaternion &q_wr);

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
  Quaternion direction_wr;     // 4
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

struct PartiallyInitializedPoint {
  static constexpr const int min_distance = 0; //cm
  static constexpr const int dist_resolution = 10; //cm
  static constexpr const int N_prob_segments = 50;
  static constexpr const int max_distance = dist_resolution * N_prob_segments; //cm

  Mat sigma3d;
  Mat image_patch;
  Mat ORB_descriptor;
  Vec3d position;
  Vec3d ray_direction;
  int life_duration;
  Quaternion image_normal_direction; // normal to the image plane, with sign --->img|
  std::vector<double> prob_distribution;
  PartiallyInitializedPoint(Point2i position_2d,
                            const Mat &image,
                            const StateMean &s,
                            Mat cam_intrinsic,
                            Ptr<ORB> detector,
                            bool *success);

  PartiallyInitializedPoint(KeyPoint keypt,
                            const Mat &descriptor,
                            const Mat &image,
                            const StateMean &s,
                            Mat cam_intrinsic);
};

StateMean ToSparseState(const StateMean &state_full, const vector<bool> &is_included);
Mat ToSparseMat(const Mat &full_mat, const vector<bool> &is_included);
Mat ToSparseSigma(const Mat &full_sigma, const vector<bool> &is_included);

Mat UpdateFromSparse(const Mat &old_mat,
                     const Mat &new_sparse_mat,
                     const vector<bool> &is_included);

StateMean UpdateFromSparseState(const StateMean &old_state,
                                const StateMean &new_sparse_state,
                                const vector<bool> &is_included);

struct PointStatistics{
  double traveled_distance;
  double sum_squared_reproj_distance;
  int first_frame;
  int n_observations;
  int expected_observations;
  Point3d initial_position;

  double mean_squared_detector_distance(){
    return sum_squared_reproj_distance / n_observations;
  }
};

#endif //MY_SLAM_MY_TYPES_H
