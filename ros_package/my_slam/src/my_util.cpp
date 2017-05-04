//
// Created by arqwer on 18.04.17.
//

#include "my_util.h"

#include <stdint.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "my_geometry.h"
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
  if (salt == 0) {
    switch (val) {
      case 0: return Scalar(255, 0, 0);
        break;
      case 1: return Scalar(0, 255, 0);
        break;
      case 2: return Scalar(0, 0, 255);
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
  if (cformat) {
    cout << "{" << endl;
  }
  for (int ir = 0; ir < m.rows; ++ir) {
    if (cformat) {
      cout << "{ ";
    }
    for (int ic = 0; ic < m.cols; ++ic) {
      cout << m.at<double>(ir, ic);
      if (cformat && ic != m.cols - 1) {
        cout << ",";
      }
      cout << " ";
    }
    if (cformat) {
      cout << "}";
      if (ir != m.rows - 1) {
        cout << ",";
      }
    }
    cout << endl;
  }
  if (cformat) {
    cout << "}" << endl;
  }
}

void DrawCross(Mat output_mat, Point2d pt, Scalar color, int size) {
  Point2d s1 = Point2d(size, size);
  Point2d s2 = Point2d(size, -size);
  line(output_mat, pt - s1, pt + s1, color, 2);
  line(output_mat, pt - s2, pt + s2, color, 2);
  line(output_mat, pt - s1, pt + s1, Scalar(255, 255, 255), 1);
  line(output_mat, pt - s2, pt + s2, Scalar(255, 255, 255), 1);
}

void DrawPlus(Mat output_mat, Point2d pt, Scalar color, int size) {
  Point2d s1 = Point2d(size, 0);
  Point2d s2 = Point2d(0, size);
  line(output_mat, pt - s1, pt + s1, color, 2);
  line(output_mat, pt - s2, pt + s2, color, 2);
  line(output_mat, pt - s1, pt + s1, Scalar(255, 255, 255), 1);
  line(output_mat, pt - s2, pt + s2, Scalar(255, 255, 255), 1);
}

void DrawPoints(Mat &output_image,
                const vector<Point2d> &points_coords,
                char marker_type, int size) {
  for (int i_obs = 0; i_obs < points_coords.size(); ++i_obs) {
    Scalar color = hashcolor(i_obs);
    if (marker_type == 'o') {
      circle(output_image, points_coords[i_obs], 4, color, 7);
    }
    if (marker_type == 'c') {
      circle(output_image, points_coords[i_obs], size, color, 1);
    }
    if (marker_type == 'x') {
      DrawCross(output_image, points_coords[i_obs], color, size);
    }
    if (marker_type == '+') {
      DrawPlus(output_image, points_coords[i_obs], color, size);
    }
  }
}

Mat ImageFromMsg(const sensor_msgs::ImageConstPtr &msg) {
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
                                             const vector<Point2d> &expected_positions,
                                             int search_radius,
                                             const NormTypes &norm_type) {
  vector<Point2d> coordinates_vec;
  for (int i_known_kp = 0; i_known_kp < known_descriptors.rows; ++i_known_kp) {
    double min_descr_distance =
        norm(known_descriptors.row(i_known_kp), kp_descriptors.row(0), norm_type);
    int closest_id = -1;
    for (int i_kp = 0; i_kp < key_points.size(); ++i_kp) {
      double pixel_distance = norm(expected_positions[i_known_kp] - Point2d(key_points[i_kp].pt));
      if (pixel_distance > search_radius) {
        continue;
      }
      double descr_distance =
          norm(known_descriptors.row(i_known_kp), kp_descriptors.row(i_kp), norm_type);
      if (descr_distance < min_descr_distance) {
        min_descr_distance = descr_distance;
        closest_id = i_kp;
      }
    }
    if(closest_id == -1){
      coordinates_vec.push_back(expected_positions[i_known_kp]); //Todo: think and fix
    }else{
      coordinates_vec.push_back(key_points[closest_id].pt);
    }
  }
  return coordinates_vec;
}

double mod(double a, double b) {
  while (a > b) {
    a -= b;
  }
  return a;
}

double limitPi(double a) {
  a = mod(a, pi * 2);
  if (a > pi) {
    a -= 2 * pi;
  }
  return a;
}

double NormalPdf2d(const Mat &sigma, Vec2d mean, Vec2d x) {
  //Todo: test
  Mat xMinusMu(x - mean);
  Mat expParameter = -1 / 2.0 * xMinusMu.t() * sigma.inv() * xMinusMu;
  return pow(2 * pi, -1 / 2.0 * 2) * pow(determinant(sigma), -1 / 2.0)
      * exp(expParameter.at<double>(0, 0));
}

Mat L2DistanceMat(const Mat &image, const Mat &patch) {
  int res_width = image.cols - patch.cols;
  int res_height = image.rows - patch.rows;
  Mat result(res_height, res_width, CV_64F);
  for (int x = 0; x < res_width; x++) {
    for (int y = 0; y < res_height; y++) {
      Rect match(x, y, patch.cols, patch.rows);
      result.at<double>(y, x) = norm(image(match), patch);
    }
  }
  return result;
}

Point2i ArgMin(Mat values) {
  double minval = values.at<double>(0, 0);
  Point2i min_pt(0, 0);
  for (int row = 0; row < values.rows; ++row) {
    for (int col = 0; col < values.cols; ++col) {
      if (values.at<double>(row, col) < minval) {
        minval = values.at<double>(row, col);
        min_pt = Point2i(col, row);
      }
    }
  }
  return min_pt;
}

Point2i L2MatchingPosition(const Mat &image, const Mat &patch, Rect search_region) {
  int x = 0;
  int y = 0;
  Point2i best;
  double least_distance = std::numeric_limits<double>::infinity();
  int search_width = search_region.width;
  int search_height = search_region.height;
  const int x0 = search_region.x - patch.cols / 2;
  const int y0 = search_region.y - patch.rows / 2;
  for (x = max(x0, 0); x < min(x0 + search_width, image.cols - patch.cols); ++x) {
    for (y = max(y0, 0); y < min(y0 + search_height, image.rows - patch.rows); ++y) {
      Rect match(x, y, patch.cols, patch.rows);
      double distance = norm(image(match), patch);
      if (distance < least_distance) {
        least_distance = distance;
        best.x = x + patch.cols / 2;
        best.y = y + patch.rows / 2;
      }
    }
  }
  return best;
}

void normalize(vector<double> *vec) {
  double divisor = norm(*vec);
  for (int i = 0; i < vec->size(); i++) {
    (*vec)[i] = (*vec)[i] / divisor;
  }
}

void normalize(vector<double> *vec, double divisor) {
  for (int i = 0; i < vec->size(); i++) {
    (*vec)[i] = (*vec)[i] / divisor;
  }
}


Mat CovarianceAlongLine(double x, double y, double z, double dispersion, double perp_dispersion) {
  double sigma_array[3][3] = {{dispersion, 0, 0}, {0, perp_dispersion, 0}, {0, 0, perp_dispersion}};
  Mat sigma(3, 3, CV_64F, &sigma_array);
  Mat R = RotationMatXtoXYZ(x, y, z);
  Mat res = R * sigma * R.t();
  return res.clone();
}

void draw_covariance_mat_ellipse(){

}