//
// Created by arqwer on 18.04.17.
//

#include "my_types.h"
#include "my_util.h"
#include "my_geometry.h"
#include "opencv2/opencv.hpp"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

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

Quaternion operator*(const Quaternion &q, double val) {
  return Quaternion(q.w() * val, q.i() * val, q.j() * val, q.k() * val);
}
Quaternion operator*(double val, const Quaternion &q) {
  return q * val;
}
Quaternion operator/(const Quaternion &q, double val) {
  return q * (1.0 / val);
}

Mat state2mat(StateMean s) {
  int size = N_VARS_FOR_CAMERA + 3 * s.feature_positions_w.size();
  Mat result(size, 1, CV_64F);
  result.at<double>(0, 0) = s.position_w.x;
  result.at<double>(1, 0) = s.position_w.y;
  result.at<double>(2, 0) = s.position_w.z;

  result.at<double>(3, 0) = s.direction_wr.w();
  result.at<double>(4, 0) = s.direction_wr.i();
  result.at<double>(5, 0) = s.direction_wr.j();
  result.at<double>(6, 0) = s.direction_wr.k();

  result.at<double>(7, 0) = s.velocity_w.x;
  result.at<double>(8, 0) = s.velocity_w.y;
  result.at<double>(9, 0) = s.velocity_w.z;
  result.at<double>(10, 0) = s.angular_velocity_r.x;
  result.at<double>(11, 0) = s.angular_velocity_r.y;
  result.at<double>(12, 0) = s.angular_velocity_r.z;
  for (int i_pt = 0; i_pt < s.feature_positions_w.size(); ++i_pt) {
    result.at<double>(N_VARS_FOR_CAMERA + i_pt * 3 + 0, 0) = s.feature_positions_w[i_pt].x;
    result.at<double>(N_VARS_FOR_CAMERA + i_pt * 3 + 1, 0) = s.feature_positions_w[i_pt].y;
    result.at<double>(N_VARS_FOR_CAMERA + i_pt * 3 + 2, 0) = s.feature_positions_w[i_pt].z;
  }
  return result;
}

StateMean::StateMean(Mat m) {
  assert(m.rows >= N_VARS_FOR_CAMERA);
  int n_points = (m.rows - N_VARS_FOR_CAMERA) / 3;
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
  direction_wr = Quaternion(qw, qi, qj, qk);
  velocity_w = Point3d(vx, vy, vz);
  angular_velocity_r = Point3d(wx, wy, wz);
  for (int i_pt = 0; i_pt < n_points; ++i_pt) {
    double px = m.at<double>(N_VARS_FOR_CAMERA + i_pt * 3 + 0, 0);
    double py = m.at<double>(N_VARS_FOR_CAMERA + i_pt * 3 + 1, 0);
    double pz = m.at<double>(N_VARS_FOR_CAMERA + i_pt * 3 + 2, 0);
    feature_positions_w.push_back(Point3d(px, py, pz));
  }
}
ostream &operator<<(ostream &os, const StateMean &stateMean) {
  os << "position_w:\t" << stateMean.position_w << endl
     << "direction_wr:\t" << stateMean.direction_wr << " phi = "
     << limitPi(acos(stateMean.direction_wr.w()) * 2) << endl
     << "velocity_w:\t" << stateMean.velocity_w << endl
     << "angular_velocity_r:\t" << stateMean.angular_velocity_r << " phi = "
     << limitPi(norm(stateMean.angular_velocity_r)) << endl
     << "feature_positions_w:\n" << stateMean.feature_positions_w;
  return os;
}

double norm(const Quaternion &q) {
  return sqrt(q.w() * q.w() + q.i() * q.i() + q.j() * q.j() + q.k() * q.k());
}

void StateToMsg(const StateMean &s, vector<Point3d> trajectory, PointCloud *points3D, const vector<PointStatistics> &pt_stats) {
  pcl::PointXYZRGB camera_pt(255, 255, 255);
  camera_pt.x = s.position_w.x;
  camera_pt.y = s.position_w.z;
  camera_pt.z = -s.position_w.y;
  points3D->push_back(camera_pt);

  for (int i_traj_pt = 0; i_traj_pt < trajectory.size(); ++i_traj_pt) {
//    Scalar pt_color = hashcolor(i_traj_pt,1);
    double brightness = 1.0 * i_traj_pt / trajectory.size() * 255.0;
    pcl::PointXYZRGB point3d(brightness, brightness, brightness * 0.8 + 0.2 * 255);
    point3d.x = trajectory[i_traj_pt].x;
    point3d.y = trajectory[i_traj_pt].z;
    point3d.z = -trajectory[i_traj_pt].y;
    points3D->push_back(point3d);
  }

  for (int i_map_pt = 0; i_map_pt < s.feature_positions_w.size(); ++i_map_pt) {
    Scalar pt_color = hashcolor(pt_stats[i_map_pt].uid);
    pcl::PointXYZRGB point3d(pt_color[2], pt_color[1], pt_color[0]);
    point3d.x = s.feature_positions_w[i_map_pt].x;
    point3d.y = s.feature_positions_w[i_map_pt].z;
    point3d.z = -s.feature_positions_w[i_map_pt].y;
    points3D->push_back(point3d);
  }

}

Vec3d RotateByQuat(const Quaternion &q, const Vec3d &v) {
  Quaternion resq = q * Vec2Quat(v) * conj(q);
  return Vec3d(resq.i(), resq.j(), resq.k());
}

Vec3d Direction(const Quaternion &q_wr) {
  return RotateByQuat(q_wr, Vec3d(0, 0, 1));
}

Quaternion QuaternionFromAxisAngle(Point3d axis_angle) {
  Quaternion res;
  double theta = norm(axis_angle);
  if (theta < 0.0000000001) {
    return Quaternion(1, 0, 0, 0);
  }
  Point3d unit_vec = axis_angle / theta;
  res.w() = cos(theta / 2);
  res.i() = unit_vec.x * sin(theta / 2);
  res.j() = unit_vec.y * sin(theta / 2);
  res.k() = unit_vec.z * sin(theta / 2);
  return res;
}

Quaternion conj(const Quaternion &q) {
  return Quaternion{q.w(), -q.i(), -q.j(), -q.k()};
}

Quaternion Vec2Quat(const Vec3d &v) {
  return Quaternion{0, v[0], v[1], v[2]};
}

PartiallyInitializedPoint::PartiallyInitializedPoint(Point2i position_2d,
                                                     const Mat &image,
                                                     const StateMean &s,
                                                     Mat cam_intrinsic,
                                                     Ptr<ORB> detector,
                                                     bool *success) :
    prob_distribution(N_prob_segments, 1.0f / N_prob_segments) {
  life_duration = 10;
  position = s.position_w;
  Vec3d rel_ray = RayFromXY_rel(position_2d.x, position_2d.y, cam_intrinsic);
  ray_direction = RotateByQuat(s.direction_wr, rel_ray);
  image_normal_direction =
      s.direction_wr; // Todo: change to direction from MonoSLAM (ortogonal to line)
  int vicinity_w = 11;
  int vicinity_h = 11;
  Rect vicinity
      {position_2d.x - vicinity_w / 2,
       position_2d.y - vicinity_h / 2,
       vicinity_w,
       vicinity_h};
  image_patch = image(vicinity).clone();

  int bigger_vicinity_w = 33;
  int bigger_vicinity_h = 33;
  Rect bigger_vicinity
      {position_2d.x - bigger_vicinity_w / 2,
       position_2d.y - bigger_vicinity_h / 2,
       bigger_vicinity_w,
       bigger_vicinity_h};
  Mat bigger_patch = image(bigger_vicinity).clone();
  //TODO: remove orb detector from here
//  int nfeatures = 50; //Todo: change to propper value
//  float scaleFactor = 1.2f;
//  int nlevels = 8;
//  int edgeThreshold = 15; // Changed default (31);
//  int firstLevel = 0;
//  int WTA_K = 2;
//  int scoreType = ORB::HARRIS_SCORE;
//  int patchSize = 31;
//  int fastThreshold = 20;
//
//  Ptr<ORB> detector = ORB::create(
//      nfeatures,
//      scaleFactor,
//      nlevels,
//      edgeThreshold,
//      firstLevel,
//      WTA_K,
//      scoreType,
//      patchSize,
//      fastThreshold
//  );
  Mat descriptors;
  std::vector<KeyPoint> keypoints;
  detector->detectAndCompute(bigger_patch, noArray(), keypoints, descriptors);
  if (keypoints.size() <= 0) {
    *success = false;
    return;
  }
  *success = true;
  ORB_descriptor = descriptors.row(0).clone();
  sigma3d = CovarianceAlongLine(ray_direction[0],
                                ray_direction[1],
                                ray_direction[2],
                                2 * dist_resolution, // todo: calculate properly
                                0.02 * dist_resolution);
}

PartiallyInitializedPoint::PartiallyInitializedPoint(KeyPoint keypt,
                                                     const Mat &descriptor,
                                                     const Mat &image,
                                                     const StateMean &s,
                                                     Mat cam_intrinsic) :
    prob_distribution(N_prob_segments, 1.0f / N_prob_segments) {
  Point2i position_2d = keypt.pt;
  life_duration = 10;
  position = s.position_w;
  Vec3d rel_ray = RayFromXY_rel(position_2d.x, position_2d.y, cam_intrinsic);
  ray_direction = RotateByQuat(s.direction_wr, rel_ray);
  image_normal_direction =
      s.direction_wr; // Todo: change to direction from MonoSLAM (ortogonal to line)
  int vicinity_w = 11;
  int vicinity_h = 11;
  Rect vicinity
      {position_2d.x - vicinity_w / 2,
       position_2d.y - vicinity_h / 2,
       vicinity_w,
       vicinity_h};
  image_patch = image(vicinity).clone();

  int bigger_vicinity_w = 33;
  int bigger_vicinity_h = 33;
  Rect bigger_vicinity
      {position_2d.x - bigger_vicinity_w / 2,
       position_2d.y - bigger_vicinity_h / 2,
       bigger_vicinity_w,
       bigger_vicinity_h};
  Mat bigger_patch = image(bigger_vicinity).clone();

  ORB_descriptor = descriptor;
  sigma3d = CovarianceAlongLine(ray_direction[0],
                                ray_direction[1],
                                ray_direction[2],
                                2 * dist_resolution, // todo: calculate properly
                                0.02 * dist_resolution);
}


StateMean ToSparseState(const StateMean &state_full, const vector<bool> &is_included) {
  assert(state_full.feature_positions_w.size() == is_included.size());
  StateMean state_sparse = state_full;
  state_sparse.feature_positions_w.resize(0);
  for (int i_kp = 0; i_kp < state_full.feature_positions_w.size(); ++i_kp) {
    if (is_included[i_kp]) {
      state_sparse.feature_positions_w.push_back(state_full.feature_positions_w[i_kp]);
    }
  }
  return state_sparse;
}



StateMean UpdateFromSparseState(const StateMean &old_state,
                                const StateMean &new_sparse_state,
                                const vector<bool> &is_included) {
  assert(old_state.feature_positions_w.size() == is_included.size());
  StateMean new_full_state = new_sparse_state;
  new_full_state.feature_positions_w = old_state.feature_positions_w;
  for (int i_kp_full = 0, i_kp_sparse = 0; i_kp_full < old_state.feature_positions_w.size();
       ++i_kp_full) {
    if (is_included[i_kp_full]) {
      new_full_state.feature_positions_w[i_kp_full] =
          new_sparse_state.feature_positions_w[i_kp_sparse++];
    }
  }
  return new_full_state;
}


Mat ToSparseMat(const Mat &full_mat, const vector<bool> &is_included) {
  assert(full_mat.cols == full_mat.rows);
  assert(full_mat.cols == is_included.size());
  int sparse_size = 0;
  for (int i = 0; i < is_included.size(); ++i) {
    if (is_included[i]) {
      ++sparse_size;
    }
  }
  Mat sparse_mat(sparse_size, sparse_size, CV_64F, double(0));
  for (int row_full = 0, row_sparse = 0; row_full < full_mat.rows; ++row_full) {
    if (is_included[row_full]) {
      for (int col_full = 0, col_sparse = 0; col_full < full_mat.cols; ++col_full) {
        if (is_included[col_full]) {
          sparse_mat.at<double>(row_sparse, col_sparse) = full_mat.at<double>(row_full, col_full);
          ++col_sparse;
        }
      }
      ++row_sparse;
    }
  }
  return sparse_mat;
}

Mat ToSparseSigma(const Mat &full_sigma, const vector<bool> &is_included) {
  vector<bool> is_colrow_included(N_VARS_FOR_CAMERA, true);
  for (int i_pt = 0; i_pt < is_included.size(); ++i_pt) {
    if (is_included[i_pt]) {
      for (int i_coord = 0; i_coord < 3; ++i_coord) {
        is_colrow_included.push_back(true);
      }
    } else {
      for (int i_coord = 0; i_coord < 3; ++i_coord) {
        is_colrow_included.push_back(false);
      }
    }
  }
  Mat sparse_Sigma_state_cov = ToSparseMat(full_sigma, is_colrow_included);
  return sparse_Sigma_state_cov;
}

Mat UpdateFromSparse(const Mat &old_mat,
                     const Mat &new_sparse_mat,
                     const vector<bool> &is_included) {

  assert(new_sparse_mat.rows == new_sparse_mat.cols);
  assert(old_mat.rows == old_mat.cols);
  assert(is_included.size() == old_mat.cols);
  int sparse_size = 0;
  for (int i = 0; i < is_included.size(); ++i) {
    if (is_included[i]) {
      ++sparse_size;
    }
  }
  assert(sparse_size == new_sparse_mat.cols);

  Mat new_full_mat = old_mat.clone();
  for (int row_full = 0, row_sparse = 0; row_full < old_mat.rows; ++row_full) {
    if (is_included[row_full]) {
      for (int col_full = 0, col_sparse = 0; col_full < old_mat.cols; ++col_full) {
        if (is_included[col_full]) {
          new_full_mat.at<double>(row_full, col_full) = new_sparse_mat.at<double>(row_sparse, col_sparse);
          ++col_sparse;
        }
      }
      ++row_sparse;
    }
  }
  return new_full_mat;
}

