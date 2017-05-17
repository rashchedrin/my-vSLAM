#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "my_util.h"
#include "my_types.h"
#include "jacobians.h"
#include "my_geometry.h"


//#include <pcl_ros/point_cloud.h>
//#include <pcl/point_types.h>

auto &&dbg = cout;

using namespace cv;
using namespace std;

static const std::string OPENCV_WINDOW = "Image window 1";

template<typename MAT_TYPE>
uint64_t mhash(Mat A) {
  uint64_t result = 0;
  uint64_t prime = 19997;
  uint64_t multiplier = prime;
  for (int i_row = 0; i_row < A.rows; ++i_row) {
    for (int i_col = 0; i_col < A.cols; ++i_col) {
      auto value = A.at<MAT_TYPE>(i_row, i_col);
      result += value * multiplier;
      multiplier *= prime;
      if (multiplier == 0) {
        ++multiplier;
      }
    }
  }
  return result;
}

StateMean predict_state(const StateMean &s0,
                        Point3d delta_vel,
                        Point3d delta_angular_vel,
                        double delta_time) {
  StateMean result = s0;
  result.position_w = s0.position_w + delta_time * (s0.velocity_w + delta_vel);
  result.direction_wr =
      s0.direction_wr
          * QuaternionFromAxisAngle(delta_time * (s0.angular_velocity_r + delta_angular_vel));
  result.velocity_w = s0.velocity_w + delta_vel;
  result.angular_velocity_r = s0.angular_velocity_r + delta_angular_vel;
  return result;
}

//Full matrices input
//2N x 2N
Mat S_t_innovation_cov(const Mat &H_t, const Mat &Sigma_predicted, double Q_sensor_noise) {
  Mat result = H_t * Sigma_predicted * H_t.t();
  for (int i = 0; i < result.rows; i++) {
    result.at<double>(i, i) += Q_sensor_noise;
  }
  return result;
}

vector<Point2d> FeaturesProjections(const StateMean &s, const Mat &camIntrinsics) {
  vector<Point2d> result;
  for (int i_pt = 0; i_pt < s.feature_positions_w.size(); ++i_pt) {
    Point2d projection;
    projection =
        ProjectPoint(s.feature_positions_w[i_pt], s.position_w, s.direction_wr, camIntrinsics);
    result.push_back(projection);
  }
  return result;
}

//13 x 13
Mat predict_Sigma_cam(const Mat &Sigma_cam,
                      const StateMean &s,
                      double delta_time,
                      const Mat &Pn_noise_cov) {
  Mat F = Ft_df_over_dxcam(s, delta_time);
  Mat Q = Q_df_over_dn(s, delta_time);
  return F * Sigma_cam * F.t() + Q * Pn_noise_cov * Q.t();
}

Mat predict_Sigma_full(const Mat &Sigma_full,
                       const StateMean &state,
                       double delta_time,
                       const Mat &Pn_noise_cov) {
  Mat result = Sigma_full.clone();
  Mat Sigma_cam_area = result(Rect(0, 0, N_VARS_FOR_CAMERA, N_VARS_FOR_CAMERA));
  Mat Sigma_cam_pred = predict_Sigma_cam(Sigma_cam_area, state, delta_time, Pn_noise_cov);
  Sigma_cam_pred.copyTo(Sigma_cam_area);
  return result;
}

//3N+13 x 2N
Mat Kalman_Gain(const Mat &H_t,
                const StateMean &state,
                const Mat &Pn_noise_cov,
                const Mat &Sigma_predicted, const Mat &innovation_cov) {
  return Sigma_predicted * H_t.t() * innovation_cov.inv();
}

class MY_SLAM {

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;

  image_transport::Subscriber image_sub_;
  ros::Publisher pub;

  StateMean x_state_mean;
  Mat Sigma_state_cov;
  vector<PointStatistics> pt_statistics;

  Mat Pn_noise_cov; // 6 x 6
  Mat current_frame;

  vector<Point2d> obstacles_for_new_points;
  Ptr<ORB> ORB_detector;
  vector<Point3d> trajectory;
  Mat output_mat;
  Mat innovation_cov;

  int frame_number = 0;

  list <PartiallyInitializedPoint> partially_initialized_pts;
//<editor-fold desc="descriptors">
  unsigned char known_descriptors_array_ORB_HD[4][32] = {
      {64, 188, 154, 117, 16, 108, 49, 23, 186, 187, 46, 24, 182, 179, 64, 26, 129, 118, 104, 16,
       50, 125, 147, 3, 205, 167, 52, 114, 192, 64, 7, 43},
      {89, 73, 7, 165, 16, 198, 47, 164, 164, 192, 194, 64, 245, 23, 33, 226, 96, 229, 120, 238, 73,
       153, 145, 45, 177, 162, 238, 0, 40, 50, 33, 40},
      {219, 206, 193, 23, 186, 48, 79, 182, 49, 199, 142, 244, 239, 39, 130, 214, 39, 247, 253, 140,
       104, 205, 229, 107, 186, 246, 254, 144, 162, 244, 172, 125},
      {240, 173, 90, 245, 212, 200, 109, 231, 49, 39, 226, 26, 55, 35, 64, 17, 41, 159, 67, 136,
       122, 158, 128, 111, 253, 132, 150, 97, 33, 0, 230, 170}
  };

  Mat known_ORB_descriptors = Mat(4, 32, CV_8UC1, &known_descriptors_array_ORB_HD).clone();

//</editor-fold>
  //calibrated
  double camera_intrinsic_array[3][3] = {{-1.0166722592048205e+03, 0., 6.3359083662958744e+02},
                                         {0., -1.0166722592048205e+03, 3.5881262886802512e+02},
                                         {0., 0., 1.}};

  //artificial, with principle point in center
  double camera_intrinsic_array2[3][3] = {{-1.0166722592048205e+03, 0., 612.105},
                                          {0., -1.0166722592048205e+03, 375.785},
                                          {0., 0., 1.}};

  const Mat camera_intrinsic = Mat(3, 3, CV_64F, &camera_intrinsic_array).clone();

  bool IsStateCorrect() {
    if (isnan(x_state_mean.position_w.x) ||
        isnan(x_state_mean.position_w.y) ||
        isnan(x_state_mean.position_w.z)) {
      cout << "Lost location" << endl;
      return false;
    }

    if (abs(norm(x_state_mean.direction_wr) - 1) > 0.5) {
      cout << "Lost direction" << endl;
      cout << x_state_mean.direction_wr << endl;
      cout << norm(x_state_mean.direction_wr) << endl;
      return false;
    }

    if (norm(x_state_mean.position_w) > 500) {
      cout << "Position is too far" << endl;
      cout << x_state_mean.position_w << endl;
      cout << norm(x_state_mean.position_w) << endl;
      return false;
    }
    return true;
  }

  Rect GetRandomInterestingRectangle(int padding = 15, int width = 150, int height = 100) {
    //Properties:
    // [V] 1. Doesn't contain any already-known features
    // [ ] 2. Will not disappear immediately //Todo
    // [V] 3. Random

    Rect res;
    int img_w = current_frame.cols;
    int img_h = current_frame.rows;
    int x_min = padding;
    int x_max = img_w - padding - width;
    int y_min = padding;
    int y_max = img_h - padding - height;
    bool intersects_features = false;
    int counter = 0;
    int max_iter = 20;
    int x, y;
    do {
      // Todo: use std::rand
      x = rand() % (x_max - x_min) + x_min;
      y = rand() % (y_max - y_min) + y_min;
      for (auto &&proj : obstacles_for_new_points) {
        if (proj.x > x - padding &&
            proj.y > y - padding &&
            proj.x < x + width + padding &&
            proj.y < y + height + padding) {
          intersects_features = true;
        }
      }
      counter++;
      if (counter > max_iter) {
        return Rect(-1, -1, 0, 0);
      }
    } while (intersects_features);
    return Rect(x, y, width, height);
  }

  Point2i FindShiTomasiMax(Mat image) {
    int maxCorners = 1;

    /// Parameters for Shi-Tomasi algorithm
    //Todo: optimize
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    //Todo: think about useHarrisDetector = ?
    bool useHarrisDetector = false;
    double k = 0.04;

    /// Copy the source image
    Mat image_gray;
    cvtColor(image, image_gray, CV_BGR2GRAY);
    /// Apply corner detection
    goodFeaturesToTrack(image_gray,
                        corners,
                        maxCorners,
                        qualityLevel,
                        minDistance,
                        Mat(),
                        blockSize,
                        useHarrisDetector,
                        k);
    if (corners.size() < 1) {
      return Point2i(-1, -1);
    }
    return Point2i(corners[0]);
  }
  void DetectNewFeatures(Mat image) {
    static int call_counter = 0;
    call_counter++;
    Rect roi = GetRandomInterestingRectangle();
    if (roi.height == 0) {
      return;
    }
    Point2i new_candidate = FindShiTomasiMax(image(roi)) + Point2i(roi.x, roi.y);
    if (new_candidate.x < 0) {
      return;
    }
    bool success;
    auto point = PartiallyInitializedPoint(new_candidate,
                                           current_frame,
                                           x_state_mean,
                                           camera_intrinsic,
                                           ORB_detector,
                                           &success);
    if (!success) {
      return;
    }
    partially_initialized_pts.push_back(point);
    DrawCross(output_mat, new_candidate, Scalar(255, 255, 255), 10);
  }

  void DetectNewFeatures2(Mat image) {
    static int call_counter = 0;
    call_counter++;
    Mat descriptors;
    std::vector<KeyPoint> keypoints;
    //Todo: run detectAndCompute only once
    ORB_detector->detectAndCompute(current_frame, noArray(), keypoints, descriptors);

    double NEW_PT_MIN_DISTANCE = 50;
    int MIN_X = 65;
    int MIN_Y = 65;
    int MAX_X = current_frame.cols - 65;
    int MAX_Y = current_frame.rows - 65;
    double max_response = -std::numeric_limits<double>::infinity();
    int max_i = -1;
    for (int i_kp = 0; i_kp < keypoints.size(); ++i_kp) {
      if (keypoints[i_kp].response > max_response) {
        if (
            keypoints[i_kp].pt.x < MIN_X ||
                keypoints[i_kp].pt.y < MIN_Y ||
                keypoints[i_kp].pt.x > MAX_X ||
                keypoints[i_kp].pt.y > MAX_Y
            ) {
          continue;
        }
        bool interesting = true;
        for (int i_obst = 0; i_obst < obstacles_for_new_points.size(); i_obst++) {
          if (norm(Point2d{obstacles_for_new_points[i_obst].x - keypoints[i_kp].pt.x,
                           obstacles_for_new_points[i_obst].y - keypoints[i_kp].pt.y})
              < NEW_PT_MIN_DISTANCE) {
            interesting = false;
            break;
          }
        }
        if (interesting) {
          max_response = keypoints[i_kp].response;
          max_i = i_kp;
        }
      }
    }

    if (max_i < 0) {
      return;
    }

    auto point = PartiallyInitializedPoint(keypoints[max_i],
                                           descriptors.row(max_i),
                                           current_frame,
                                           x_state_mean,
                                           camera_intrinsic);

    partially_initialized_pts.push_back(point);
    DrawCross(output_mat, keypoints[max_i].pt, Scalar(255, 255, 255), 10);
  }

  void ShowRay(Vec3d origin, Vec3d direction, double resolution, int n_segments, Scalar color) {
    for (int i_segment = 0; i_segment < n_segments; ++i_segment) {
      Point3d pos3d = origin + resolution * i_segment * direction;
      Point3d s1 = Point3d(1, 0, 0);
      Point3d s2 = Point3d(0, 1, 0);
      Point2d coord_2d_ray1 =
          ProjectPoint(pos3d + s1 + s2,
                       x_state_mean.position_w,
                       x_state_mean.direction_wr,
                       camera_intrinsic);
      Point2d coord_2d_ray2 =
          ProjectPoint(pos3d + s1 - s2,
                       x_state_mean.position_w,
                       x_state_mean.direction_wr,
                       camera_intrinsic);
      Point2d coord_2d_ray3 =
          ProjectPoint(pos3d - s1 - s2,
                       x_state_mean.position_w,
                       x_state_mean.direction_wr,
                       camera_intrinsic);
      Point2d coord_2d_ray4 =
          ProjectPoint(pos3d - s1 + s2,
                       x_state_mean.position_w,
                       x_state_mean.direction_wr,
                       camera_intrinsic);
      line(output_mat, coord_2d_ray1, coord_2d_ray2, color);
      line(output_mat, coord_2d_ray2, coord_2d_ray3, color);
      line(output_mat, coord_2d_ray3, coord_2d_ray4, color);
      line(output_mat, coord_2d_ray4, coord_2d_ray1, color);
    }
  }

  void UpdatePartiallyInitializedFeaturs(const Mat &image) {
    int i_pt = 0;
    for (auto it_pt = partially_initialized_pts.begin();
         it_pt != partially_initialized_pts.end();) {
      auto &&pt = *it_pt;
      bool remove_this_point = false;
      Point2d origin_2d = ProjectPoint(pt.position,
                                       x_state_mean.position_w,
                                       x_state_mean.direction_wr,
                                       camera_intrinsic);
      ShowRay(pt.position,
              pt.ray_direction,
              pt.dist_resolution,
              pt.N_prob_segments,
              hashcolor(i_pt + known_ORB_descriptors.rows));
      DrawCross(output_mat, origin_2d, Scalar(0, 255, 255), 5);

//      Mat distances = L2DistanceMat(image,pt.image);
      Mat distances
          (image.rows - pt.image_patch.rows + 1, image.cols - pt.image_patch.cols + 1, CV_32FC1);
      matchTemplate(image, pt.image_patch, distances, CV_TM_SQDIFF);
      vector<bool> is_updated(pt.N_prob_segments, true);
      for (int i_segment = 0; i_segment < pt.N_prob_segments; ++i_segment) {
        Point3d pos3d = pt.position + pt.dist_resolution * i_segment * pt.ray_direction;

        Point2d coord_2d =
            ProjectPoint(pos3d,
                         x_state_mean.position_w,
                         x_state_mean.direction_wr,
                         camera_intrinsic);
//        Scalar color = hashcolor(i_pt,2);
        Mat dh_over_dx = Dh_pt_over_dx_cam(x_state_mean, camera_intrinsic, pos3d); // 2 x 7
        Mat sigma_cam = Sigma_state_cov(Rect(0, 0, 7, 7));
        Mat dh_over_dz = Dh_pt_over_dz(x_state_mean, camera_intrinsic, pos3d); //2 x 3
        Mat pixel_noise_cov = Mat::eye(2, 2, CV_64F) * pixel_noise;
        Mat sigma =
            dh_over_dx * sigma_cam * dh_over_dx.t() + dh_over_dz * pt.sigma3d * dh_over_dz.t()
                + pixel_noise_cov;
        double probability_threshold = 1.5; // sigmas
        double search_height = sigma.at<double>(0, 0) * probability_threshold;
        double search_width = sigma.at<double>(1, 1) * probability_threshold;
//        dbg<<search_width<<" x "<<search_height<<endl;
        Rect center_search_region(coord_2d.x - search_width / 2,
                                  coord_2d.y - search_height / 2,
                                  search_width,
                                  search_height);
        int argmin_x_from = center_search_region.x - pt.image_patch.cols / 2;
        int argmin_y_from = center_search_region.y - pt.image_patch.rows / 2;
        int argmin_x_to = argmin_x_from + search_width;
        int argmin_y_to = argmin_y_from + search_height;
        if (argmin_x_from >= distances.cols || argmin_y_from >= distances.rows || argmin_x_to <= 0
            || argmin_y_to <= 0) {
//          pt.prob_distribution[i_segment] = 0; //TODO: think and correct
          is_updated[i_segment] = false;
          continue;
        }
        argmin_x_from = max(argmin_x_from, 0);
        argmin_y_from = max(argmin_y_from, 0);
        argmin_x_to = min(argmin_x_to, distances.cols);
        argmin_y_to = min(argmin_y_to, distances.rows);
        argmin_x_to = max(argmin_x_to, argmin_x_from);
        argmin_y_to = max(argmin_y_to, argmin_y_from);
        Rect argmin_region(argmin_x_from,
                           argmin_y_from,
                           argmin_x_to - argmin_x_from,
                           argmin_y_to - argmin_y_from);
//        rectangle(output_mat, center_search_region, hashcolor(i_pt));
//        dbg << argmin_region << endl;
        Point minLoc, maxLoc;
        double minVal, maxVal;
        if (argmin_region.width > 0 && argmin_region.height > 0) {
          minMaxLoc(distances(argmin_region), &minVal, &maxVal, &minLoc, &maxLoc);
        } else {
          minVal = distances.at<float>(argmin_region.y, argmin_region.x);
          maxVal = minVal;
          minLoc = maxLoc = Point(argmin_region.x, argmin_region.y);
        }

        Point2i best_match_pt = minLoc + Point2i(argmin_region.x, argmin_region.y)
            + Point2i(pt.image_patch.cols / 2, pt.image_patch.rows / 2);
        obstacles_for_new_points.push_back(best_match_pt);
        DrawCross(output_mat, best_match_pt, hashcolor(i_pt));
        Vec2d best_match_vec;
        best_match_vec[0] = best_match_pt.x;
        best_match_vec[1] = best_match_pt.y;
        double old_prob = pt.prob_distribution[i_segment];
//        Todo: take image diffirence norm into account
        double new_probability = old_prob * NormalPdf2d(sigma, coord_2d, best_match_vec);
        pt.prob_distribution[i_segment] = new_probability;
      }
//      normalize(&pt.prob_distribution, sum(pt.prob_distribution)[0]);

      //Normalization
      double remaining_prob = 1;
      double sum_of_updated = 0;
      for(int i_seg = 0; i_seg < pt.N_prob_segments; ++i_seg){
        if(!is_updated[i_seg]){
          remaining_prob -= pt.prob_distribution[i_seg];
        }else{
          sum_of_updated += pt.prob_distribution[i_seg];
        }
      }
      assert(remaining_prob >= 0);
      assert(sum_of_updated <= 1);
      for(int i_seg = 0; i_seg < pt.N_prob_segments; ++i_seg){
        if(is_updated[i_seg]){
          pt.prob_distribution[i_seg] *= remaining_prob / sum_of_updated;
        }
      }

      --pt.life_duration;
      if (pt.life_duration <= 0) {
        remove_this_point = true;
      }

      i_pt++;
      if (remove_this_point) {
        it_pt = partially_initialized_pts.erase(it_pt);
      } else {
        ++it_pt;
      }
    }
    ConvertToFullyInitialized();
  }

  void subscription_callback(const sensor_msgs::ImageConstPtr &msg_in) {
    assert(known_ORB_descriptors.rows == x_state_mean.feature_positions_w.size());
    assert(pt_statistics.size() == x_state_mean.feature_positions_w.size());
    static int call_counter = 0;
    int msg_number = std::stoi(msg_in->header.frame_id);
    frame_number++;
    call_counter++;
    int delta_frames = 1;
    if (frame_number != msg_number) {
      cout << "FRAMES MISSED" << endl;
      delta_frames = 1 + msg_number - frame_number;
      frame_number = msg_number;
    }

    current_frame = ImageFromMsg(msg_in);
    output_mat = current_frame.clone();
    trajectory.push_back(x_state_mean.position_w);
    obstacles_for_new_points.resize(0);

    EKF_iteration(current_frame, delta_frames);
    UpdatePartiallyInitializedFeaturs(current_frame);
    if(call_counter % 5 == 0) {
      DetectNewFeatures2(current_frame);
    }
    if(call_counter % 20 == 0 && call_counter > 40) {
//      RemoveBadPoints();
    }
    RemoveBadPoints2();


    //Publish
    PublishAll();

    //Display
    DrawAxis();
    cv::imshow("raw", current_frame);
    cv::imshow(OPENCV_WINDOW, output_mat);
    dbg << "Frame: " << frame_number << " call: " << call_counter << endl;

    //Termination
    if (!IsStateCorrect()) {
      ros::shutdown();
    }
    if (cv::waitKey(1) == 27) {
      ros::shutdown();
    }
  }

  Mat MatWithoutRow(const Mat & matIn, int row){
    // Removing a row
    Mat matOut( matIn.rows - 1, matIn.cols, matIn.type()); // Result: matIn less that one row.
    if ( row > 0 ) // Copy everything above that one row.
    {
      cv::Rect rect( 0, 0, matIn.cols, row );
      matIn( rect ).copyTo( matOut( rect ) );
    }

    if ( row < matIn.rows - 1 ) // Copy everything below that one row.
    {
      cv::Rect rect1( 0, row + 1, matIn.cols, matIn.rows - row - 1 );
      cv::Rect rect2( 0, row, matIn.cols, matIn.rows - row - 1 );
      matIn( rect1 ).copyTo( matOut( rect2 ) );
    }
    return matOut;
  }

  void RemovePoint(int point_number){
    assert(point_number >= 0);
    assert(point_number < x_state_mean.feature_positions_w.size());
    StateMean new_state;
    Mat new_Sigma;
    Mat new_known_ORB_descriptors;
    vector<bool> is_included(x_state_mean.feature_positions_w.size(), true);
    is_included[point_number] = false;
    new_state = ToSparseState(x_state_mean,is_included);
    new_Sigma = ToSparseSigma(Sigma_state_cov, is_included);
    new_known_ORB_descriptors = MatWithoutRow(known_ORB_descriptors,point_number);
    x_state_mean = new_state;
    Sigma_state_cov = new_Sigma;
    known_ORB_descriptors = new_known_ORB_descriptors;
    pt_statistics.erase(pt_statistics.begin() + point_number);
    assert(pt_statistics.size() == x_state_mean.feature_positions_w.size());
    assert(Sigma_state_cov.rows == Sigma_state_cov.cols);
    assert(pt_statistics.size() == known_ORB_descriptors.rows);
  }

  void RemoveBadPoints(){
    double most_traveled_distance = 0;
    int most_traveled_id = -1;
    double most_mean_sq_distance = 0;
    int most_sq_distance_id = -1;
    for(int i_pt = 0; i_pt < pt_statistics.size(); ++i_pt){
      if(pt_statistics[i_pt].traveled_distance > most_traveled_distance){
        most_traveled_id = i_pt;
        most_traveled_distance = pt_statistics[i_pt].traveled_distance;
      }
      if(pt_statistics[i_pt].mean_squared_detector_distance() > most_mean_sq_distance){
        most_mean_sq_distance = pt_statistics[i_pt].mean_squared_detector_distance();
        most_sq_distance_id = i_pt;
      }
    }
    assert(most_sq_distance_id != -1);
    assert(most_traveled_id != -1);
    RemovePoint(most_sq_distance_id);
  }

  void RemoveBadPoints2(){
    double sum_traveled_distance = 0;
    double sum_mean_sq_distance = 0;
    for(int i_pt = 0; i_pt < pt_statistics.size(); ++i_pt){
      sum_traveled_distance += pt_statistics[i_pt].traveled_distance;
      sum_mean_sq_distance += pt_statistics[i_pt].mean_squared_detector_distance();
    }
    double mean_traveled_distance = sum_traveled_distance / pt_statistics.size();
    double mean_mean_sq_distance = sum_mean_sq_distance / pt_statistics.size();
    for(int i_pt = 0; i_pt < pt_statistics.size(); ++i_pt){
      if(pt_statistics[i_pt].n_observations > 3) {
        if (
//            norm(pt_statistics[i_pt].initial_position - x_state_mean.feature_positions_w[i_pt]) > 30 ||
            pt_statistics[i_pt].traveled_distance > mean_traveled_distance * 2.6 ||
            pt_statistics[i_pt].mean_squared_detector_distance() > mean_mean_sq_distance * 2.3
            ) {
          RemovePoint(i_pt);
        }
      }
    }
  }

  void DrawAxis() {
    double size = 50; //cm
    Point3d origin = Point3d(0, 8, 100);
    Point2d O = ProjectPoint(origin + Point3d(0, 0, 0),
                             x_state_mean.position_w,
                             x_state_mean.direction_wr,
                             camera_intrinsic);
    Point2d X = ProjectPoint(origin + Point3d(size, 0, 0),
                             x_state_mean.position_w,
                             x_state_mean.direction_wr,
                             camera_intrinsic);
    Point2d Y = ProjectPoint(origin + Point3d(0, size, 0),
                             x_state_mean.position_w,
                             x_state_mean.direction_wr,
                             camera_intrinsic);
    Point2d Z = ProjectPoint(origin + Point3d(0, 0, size),
                             x_state_mean.position_w,
                             x_state_mean.direction_wr,
                             camera_intrinsic);
    line(output_mat, O, X, Scalar(255, 0, 0));
    line(output_mat, O, Y, Scalar(0, 255, 0));
    line(output_mat, O, Z, Scalar(0, 0, 255));
  }

  void ConvertToFullyInitialized() {
    for (auto pt_it = partially_initialized_pts.begin();
         pt_it != partially_initialized_pts.end();) {
      bool remove_this_pt = false;
      int prob_argmax = 0;
      double max_prob = 0;
      for (int i_prob = 1; i_prob < (*pt_it).prob_distribution.size(); ++i_prob) {
        double prob = (*pt_it).prob_distribution[i_prob];
        if (prob > max_prob) {
          max_prob = prob;
          prob_argmax = i_prob;
        }
      }
      if (max_prob > points_init_prob_threshold) {
        dbg << "argmax: " << prob_argmax << endl;
        dbg << (*pt_it).prob_distribution[prob_argmax] << endl;
        remove_this_pt = true;
      }
      if (remove_this_pt) {
        FullyInitializePoint(*pt_it, prob_argmax);
        pt_it = partially_initialized_pts.erase(pt_it);
      } else {
        ++pt_it;
      }
    }
  }

  void FullyInitializePoint(PartiallyInitializedPoint pt, int prob_argmax) {
    Point3d pt_pos = pt.position + pt.ray_direction * prob_argmax * pt.dist_resolution;
    x_state_mean.feature_positions_w.push_back(pt_pos);
    // Add descriptor
    known_ORB_descriptors.push_back(pt.ORB_descriptor.clone());
    PointStatistics stats;
    stats.first_frame = frame_number;
    stats.n_observations = 0;
    stats.sum_squared_reproj_distance = 0;
    stats.traveled_distance = 0;
    stats.initial_position = pt_pos;
    pt_statistics.push_back(stats);
    Mat new_Sigma(Sigma_state_cov.rows + 3, Sigma_state_cov.cols + 3, CV_64F, double(0));
    Sigma_state_cov.copyTo(new_Sigma(Rect(0, 0, Sigma_state_cov.cols, Sigma_state_cov.rows)));
    pt.sigma3d.copyTo(new_Sigma(Rect(Sigma_state_cov.cols, Sigma_state_cov.rows, 3, 3)));
    Sigma_state_cov = new_Sigma.clone();
  }

  void PublishAll() const {
    PointCloud msg_out;
    static uint64_t stamp = 0;
    msg_out.header.frame_id = "map";
    msg_out.height = msg_out.width = 200;
    msg_out.header.stamp = ++stamp;
    StateToMsg(x_state_mean, trajectory, &msg_out);
    pub.publish(msg_out);
  }

  void EKF_iteration(Mat input_image, double delta_time) {
    dbg << "state:" << endl << x_state_mean << endl;
//    dbg << "Sigma:" << endl << Sigma_state_cov << endl;
//Predict
    StateMean x_state_mean_pred =
        predict_state(x_state_mean, Point3d(0, 0, 0), Point3d(0, 0, 0), delta_time);
    dbg << "predict:" << endl << x_state_mean_pred << endl;
    vector<Point2d> predicted_points = FeaturesProjections(x_state_mean_pred, camera_intrinsic);
    //yes, x_s_m
    Mat Sigma_state_cov_pred =
        predict_Sigma_full(Sigma_state_cov, x_state_mean, delta_time, Pn_noise_cov);
//Measure
    //Todo: make like in MonoSLAM, instead of ORB
    std::vector<KeyPoint> key_points;
    Mat kp_descriptors;
    ORB_detector->detectAndCompute(input_image, noArray(), key_points, kp_descriptors);

    auto &known_descriptors = known_ORB_descriptors;
    int search_radius = 45;
    //Todo: take covariance into account for search.
    vector<bool> should_be_found(predicted_points.size(), true);
    for(int i_pt = 0; i_pt < predicted_points.size(); ++i_pt){
      if(
          predicted_points[i_pt].x <= 15 ||
          predicted_points[i_pt].y <= 15 ||
          predicted_points[i_pt].x >= current_frame.cols - 15 ||
          predicted_points[i_pt].y <= current_frame.rows - 15
          ){
        should_be_found[i_pt] = false;
      }
    }
    vector<bool> is_found;
    vector<Point2d> observations = GetMatchingPointsCoordinates(key_points,
                                                                kp_descriptors,
                                                                known_descriptors,
                                                                predicted_points,
                                                                search_radius,
                                                                NORM_HAMMING,
                                                                &is_found);
    DrawPoints(output_mat, observations);
    DrawPoints(output_mat, predicted_points, 'x', 5);
    DrawPoints(output_mat, predicted_points, 'c', search_radius);
    assert(is_found.size() == x_state_mean.feature_positions_w.size());
    for (int i_pt = 0; i_pt < is_found.size(); ++i_pt) {
      if (is_found[i_pt]) {
        pt_statistics[i_pt].n_observations++;
      } else {
        if(should_be_found[i_pt]) {
          pt_statistics[i_pt].sum_squared_reproj_distance += search_radius * search_radius;
        }
      }
    }
//Update
    StateMean sparse_x_state_mean = ToSparseState(x_state_mean, is_found);
    StateMean sparse_x_state_mean_pred = ToSparseState(x_state_mean_pred, is_found);
    vector<bool> is_colrow_included(N_VARS_FOR_CAMERA, true);
    for (int i_pt = 0; i_pt < x_state_mean.feature_positions_w.size(); ++i_pt) {
      if (is_found[i_pt]) {
        for (int i_coord = 0; i_coord < 3; ++i_coord) {
          is_colrow_included.push_back(true);
        }
      } else {
        for (int i_coord = 0; i_coord < 3; ++i_coord) {
          is_colrow_included.push_back(false);
        }
      }
    }
//    is_colrow_included.insert( is_colrow_included.end(), is_found.begin(), is_found.end() );
//    cout<<"N_VARS_FOR_CAMERA: "<<N_VARS_FOR_CAMERA<<" is_found: "<<is_found.size()<<"Sigma_state_cov.cols: "<<Sigma_state_cov.cols<<endl;
//    assert(N_VARS_FOR_CAMERA + 3 * is_found.size() == Sigma_state_cov.cols);
    Mat sparse_Sigma_state_cov = ToSparseMat(Sigma_state_cov, is_colrow_included);

    Mat observations_diff =
        Mat(x_state_mean_pred.feature_positions_w.size() * 2, 1, CV_64F, double(0));
    for (int i_obs = 0; i_obs < x_state_mean_pred.feature_positions_w.size(); ++i_obs) {
      observations_diff.at<double>(i_obs * 2, 0) =
          observations[i_obs].x - predicted_points[i_obs].x;
      observations_diff.at<double>(i_obs * 2 + 1, 0) =
          observations[i_obs].y - predicted_points[i_obs].y;
    }

    static double obs_diff_accum = 0;
    static int call_count = 0;
    call_count++;
    obs_diff_accum += norm(observations_diff);
    dbg << "predict diff: " << norm(observations_diff) << endl;
    dbg << "Mean predict diff: " << obs_diff_accum / call_count << endl;
    if (norm(observations_diff) > 1000) {
      cout << "norm(observations_diff) > 1000" << endl;
      return;
    }

    //yes, x_pred
    Mat H_t = H_t_Jacobian_of_observations(x_state_mean_pred, camera_intrinsic);
    innovation_cov = S_t_innovation_cov(H_t, Sigma_state_cov_pred, pixel_noise);
    Mat KalmanGain =
        Kalman_Gain(H_t, x_state_mean_pred, Pn_noise_cov, Sigma_state_cov_pred, innovation_cov);
    Mat stateMat_pred = state2mat(x_state_mean_pred);
    StateMean x_state_mean_new = StateMean(stateMat_pred + KalmanGain * observations_diff);

    // this vv is mathematically incorrect, since EKF doesn't know about this line
    x_state_mean_new.direction_wr =
        x_state_mean_new.direction_wr / norm(x_state_mean_new.direction_wr);

    Mat Sigma_state_cov_new = (
        Mat::eye(Sigma_state_cov.rows, Sigma_state_cov.cols, CV_64F) - KalmanGain * H_t)
        * Sigma_state_cov_pred;

    for (int i_pt = 0; i_pt < x_state_mean.feature_positions_w.size(); ++i_pt) {
      pt_statistics[i_pt].traveled_distance +=
          norm(x_state_mean.feature_positions_w[i_pt] - x_state_mean_new.feature_positions_w[i_pt]);
    }
    x_state_mean = x_state_mean_new;
    Sigma_state_cov = Sigma_state_cov_new;

    vector<Point2d> reprojected_points;
    reprojected_points = FeaturesProjections(x_state_mean_new, camera_intrinsic);
    obstacles_for_new_points.insert(obstacles_for_new_points.end(), reprojected_points.begin(), reprojected_points.end());
    for (int i_pt = 0; i_pt < x_state_mean.feature_positions_w.size(); ++i_pt) {
      pt_statistics[i_pt].sum_squared_reproj_distance +=
          norm(reprojected_points[i_pt] - observations[i_pt]);
    }

    DrawPoints(output_mat, reprojected_points, '+', 5);

    dbg << endl;
    dbg << "repr, obs size:" << endl;
    dbg << reprojected_points.size() << endl;
    dbg << observations.size() << endl;
    dbg << known_descriptors.size() << endl;
//    dbg<<"Reprojection error: "<<norm(reprojected_points, observations)<<endl;
  }

  void EKF_iteration_with_sparse(Mat input_image, double delta_time) {
    dbg << "state:" << endl << x_state_mean << endl;
//    dbg << "Sigma:" << endl << Sigma_state_cov << endl;
//Predict
    StateMean x_state_mean_pred =
        predict_state(x_state_mean, Point3d(0, 0, 0), Point3d(0, 0, 0), delta_time);
    dbg << "predict:" << endl << x_state_mean_pred << endl;
    vector<Point2d> predicted_points = FeaturesProjections(x_state_mean_pred, camera_intrinsic);
    //yes, x_s_m
    Mat Sigma_state_cov_pred =
        predict_Sigma_full(Sigma_state_cov, x_state_mean, delta_time, Pn_noise_cov);
//Measure
    //Todo: make like in MonoSLAM, instead of ORB
    std::vector<KeyPoint> key_points;
    Mat kp_descriptors;
    ORB_detector->detectAndCompute(input_image, noArray(), key_points, kp_descriptors);

    auto &known_descriptors = known_ORB_descriptors;
    int search_radius = 45;
    //Todo: take covariance into account for search.
    vector<bool> should_be_found(predicted_points.size(), true);
    for(int i_pt = 0; i_pt < predicted_points.size(); ++i_pt){
      if(
          predicted_points[i_pt].x <= 15 ||
              predicted_points[i_pt].y <= 15 ||
              predicted_points[i_pt].x >= current_frame.cols - 15 ||
              predicted_points[i_pt].y <= current_frame.rows - 15
          ){
        should_be_found[i_pt] = false;
      }
    }
    vector<bool> is_found;
    vector<Point2d> observations = GetMatchingPointsCoordinates(key_points,
                                                                kp_descriptors,
                                                                known_descriptors,
                                                                predicted_points,
                                                                search_radius,
                                                                NORM_HAMMING,
                                                                &is_found);
    DrawPoints(output_mat, observations);
    DrawPoints(output_mat, predicted_points, 'x', 5);
    DrawPoints(output_mat, predicted_points, 'c', search_radius);
    assert(is_found.size() == x_state_mean.feature_positions_w.size());
    for (int i_pt = 0; i_pt < is_found.size(); ++i_pt) {
      if (is_found[i_pt]) {
        pt_statistics[i_pt].n_observations++;
      } else {
        if(should_be_found[i_pt]) {
          pt_statistics[i_pt].sum_squared_reproj_distance += search_radius * search_radius;
        }
      }
    }
//Update
    StateMean sparse_x_state_mean = ToSparseState(x_state_mean, is_found);
    if (sparse_x_state_mean.feature_positions_w.size() == 0) { //Nothing found
      x_state_mean = x_state_mean_pred;
      Sigma_state_cov = Sigma_state_cov_pred;
      return;
    }
    StateMean sparse_x_state_mean_pred = ToSparseState(x_state_mean_pred, is_found);
    vector<bool> is_colrow_included(N_VARS_FOR_CAMERA, true);
    for (int i_pt = 0; i_pt < x_state_mean.feature_positions_w.size(); ++i_pt) {
      if (is_found[i_pt]) {
        cout << "#";
        for (int i_coord = 0; i_coord < 3; ++i_coord) {
          is_colrow_included.push_back(true);
        }
      } else {
        cout << "_";
        for (int i_coord = 0; i_coord < 3; ++i_coord) {
          is_colrow_included.push_back(false);
        }
      }
    }
    cout << endl;
    Mat sparse_Sigma_state_cov = ToSparseMat(Sigma_state_cov, is_colrow_included);
    Mat sparse_Sigma_state_cov_pred = ToSparseMat(Sigma_state_cov_pred, is_colrow_included);
    auto sparse_observations = ToSparseVec(observations, is_found);
    auto sparse_predicted_points = ToSparseVec(predicted_points, is_found);
    assert(sparse_x_state_mean.feature_positions_w.size() != x_state_mean.feature_positions_w.size()
               || cv::countNonZero(sparse_Sigma_state_cov != Sigma_state_cov) == 0);
    assert(sparse_x_state_mean.feature_positions_w.size() != x_state_mean.feature_positions_w.size()
               || cv::countNonZero(sparse_Sigma_state_cov_pred != Sigma_state_cov_pred) == 0);

    Mat observations_diff =
        Mat(sparse_x_state_mean_pred.feature_positions_w.size() * 2, 1, CV_64F, double(0));
    for (int i_obs = 0; i_obs < sparse_x_state_mean_pred.feature_positions_w.size(); ++i_obs) {
      observations_diff.at<double>(i_obs * 2, 0) =
          sparse_observations[i_obs].x - sparse_predicted_points[i_obs].x;
      observations_diff.at<double>(i_obs * 2 + 1, 0) =
          sparse_observations[i_obs].y - sparse_predicted_points[i_obs].y;
    }

    static double obs_diff_accum = 0;
    static int call_count = 0;
    call_count++;
    obs_diff_accum += norm(observations_diff);
    dbg << "predict diff: " << norm(observations_diff) << endl;
    dbg << "Mean predict diff: " << obs_diff_accum / call_count << endl;
    if (norm(observations_diff) > 1000) {
      cout << "norm(observations_diff) > 1000" << endl;
      return;
    }

    //yes, x_pred
    Mat H_t = H_t_Jacobian_of_observations(sparse_x_state_mean_pred, camera_intrinsic);
    innovation_cov = S_t_innovation_cov(H_t, sparse_Sigma_state_cov_pred, pixel_noise);
    Mat KalmanGain =
        Kalman_Gain(H_t,
                    sparse_x_state_mean_pred,
                    Pn_noise_cov,
                    sparse_Sigma_state_cov_pred,
                    innovation_cov);
    Mat sparse_stateMat_pred = state2mat(sparse_x_state_mean_pred);
    StateMean
        sparse_x_state_mean_new = StateMean(sparse_stateMat_pred + KalmanGain * observations_diff);

    // this vv is mathematically incorrect, since EKF doesn't know about this line
    sparse_x_state_mean_new.direction_wr =
        sparse_x_state_mean_new.direction_wr / norm(sparse_x_state_mean_new.direction_wr);

    Mat sparse_Sigma_state_cov_new = (
        Mat::eye(sparse_Sigma_state_cov.rows, sparse_Sigma_state_cov.cols, CV_64F)
            - KalmanGain * H_t)
        * sparse_Sigma_state_cov_pred;

    StateMean x_state_mean_new =
        UpdateFromSparseState(x_state_mean_pred, sparse_x_state_mean_new, is_found);
    Mat Sigma_state_cov_new =
        UpdateFromSparse(Sigma_state_cov_pred, sparse_Sigma_state_cov_new, is_colrow_included);

    for (int i_pt = 0; i_pt < x_state_mean.feature_positions_w.size(); ++i_pt) {
      pt_statistics[i_pt].traveled_distance +=
          norm(x_state_mean.feature_positions_w[i_pt] - x_state_mean_new.feature_positions_w[i_pt]);
    }

    x_state_mean = x_state_mean_new;
    Sigma_state_cov = Sigma_state_cov_new;

    vector<Point2d> reprojected_points;
    reprojected_points = FeaturesProjections(x_state_mean_new, camera_intrinsic);
    obstacles_for_new_points.insert(obstacles_for_new_points.end(), reprojected_points.begin(), reprojected_points.end());
    for (int i_pt = 0; i_pt < x_state_mean.feature_positions_w.size(); ++i_pt) {
      pt_statistics[i_pt].sum_squared_reproj_distance +=
          norm(reprojected_points[i_pt] - observations[i_pt]);
    }
    DrawPoints(output_mat, reprojected_points, '+', 5);

    dbg << endl;
    dbg << "repr, obs size:" << endl;
    dbg << reprojected_points.size() << endl;
    dbg << observations.size() << endl;
    dbg << known_descriptors.size() << endl;
//    dbg<<"Reprojection error: "<<norm(reprojected_points, observations)<<endl;
  }
 public:
//  Predict diff:  31.6471
//  constexpr static const double pixel_noise = 4.5;
//  constexpr static const double position_speed_noise = 0.00035;
//  constexpr static const double angular_speed_noise = 0.0001;
//  constexpr static const double initial_map_uncertainty = 0;
//  constexpr static const double initial_pos_uncertainty = 45;
//  constexpr static const double initial_direction_uncertainty = 10;
//  initial_angular_velocity = 2*pi;
  constexpr static const double pixel_noise = 11;
  constexpr static const double position_speed_noise = 1;
  constexpr static const double angular_speed_noise = 0.1;
  constexpr static const double initial_map_uncertainty = 2;
  constexpr static const double initial_pos_uncertainty = 2;
  constexpr static const double initial_direction_uncertainty = 0.5;
  constexpr static const double initial_speed_uncertainty = 0.0001;
  constexpr static const double initial_angular_speed_uncertainty = 0.1;
  double points_init_prob_threshold = 0.45;
  MY_SLAM()
      : it_(nh_) {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("image_raw", 1,
                               &MY_SLAM::subscription_callback, this);
    pub = nh_.advertise<sensor_msgs::PointCloud2>("output", 1);

    Sigma_state_cov = initial_map_uncertainty * Mat::eye(25, 25, CV_64F);
    Sigma_state_cov.at<double>(0, 0) = initial_pos_uncertainty;
    Sigma_state_cov.at<double>(1, 1) = initial_pos_uncertainty;
    Sigma_state_cov.at<double>(2, 2) = initial_pos_uncertainty;
    Sigma_state_cov.at<double>(3, 3) = initial_direction_uncertainty;
    Sigma_state_cov.at<double>(4, 4) = initial_direction_uncertainty;
    Sigma_state_cov.at<double>(5, 5) = initial_direction_uncertainty;
    Sigma_state_cov.at<double>(6, 6) = initial_direction_uncertainty;
    Sigma_state_cov.at<double>(7, 7) = initial_speed_uncertainty;
    Sigma_state_cov.at<double>(8, 8) = initial_speed_uncertainty;
    Sigma_state_cov.at<double>(9, 9) = initial_speed_uncertainty;
    Sigma_state_cov.at<double>(10, 10) = initial_angular_speed_uncertainty;
    Sigma_state_cov.at<double>(11, 11) = initial_angular_speed_uncertainty;
    Sigma_state_cov.at<double>(12, 12) = initial_angular_speed_uncertainty;

//    Sigma_state_cov.at<double>(22, 22) = 1000; //todo:remove
//    Sigma_state_cov.at<double>(23, 23) = 1000;
//    Sigma_state_cov.at<double>(24, 24) = 1000;
    Pn_noise_cov = position_speed_noise * Mat::eye(6, 6, CV_64F); //todo: init properly
    Pn_noise_cov.at<double>(3, 3) = angular_speed_noise;
    Pn_noise_cov.at<double>(4, 4) = angular_speed_noise;
    Pn_noise_cov.at<double>(5, 5) = angular_speed_noise;
    //Todo: fix no-rotation == 2Pi. Really it only works if delta_time = 1
    x_state_mean.angular_velocity_r =
        Point3d(2 * pi * 0.000000000001,
                0,
                0);// no rotation, 2*pi needed to eliminate indeterminance
    x_state_mean.direction_wr = Quaternion(1, 0, 0, 0.0); // Zero rotation
    x_state_mean.position_w = Point3d(0, 0, 0);
    x_state_mean.velocity_w = Point3d(0, 0, 0);
    /* Coordinates system:
     *            /^ z
     *         /
     *      /
     *   /
     * 0------------------------>x
     * |
     * |
     * |
     * |
     * |
     * |
     * |
     * V y
     *
     * In local coordinates, camera looks from 0 to Z axis. Like here.
     */
    x_state_mean.feature_positions_w.push_back(Point3d(-40, -4, 125.3)); //cm
    x_state_mean.feature_positions_w.push_back(Point3d(-40, -41, 125.3));
    x_state_mean.feature_positions_w.push_back(Point3d(40, -41, 125.3));
    x_state_mean.feature_positions_w.push_back(Point3d(40, -4, 125.3));

    PointStatistics stat0;
    stat0.first_frame = 0; // TODO: or 1 ?
    stat0.traveled_distance = 0;
    stat0.sum_squared_reproj_distance = 0;
    stat0.n_observations = 0;
    pt_statistics.resize(4, stat0);
    for(int i = 0; i < 4; i++) {
      pt_statistics[i].initial_position = x_state_mean.feature_positions_w[i];
    }

    int nfeatures = 1000;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 15; // Changed default (31);
    int firstLevel = 0;
    int WTA_K = 2;
    int scoreType = ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;

    ORB_detector = ORB::create(
        nfeatures,
        scaleFactor,
        nlevels,
        edgeThreshold,
        firstLevel,
        WTA_K,
        scoreType,
        patchSize,
        fastThreshold
    );
  }


  ~MY_SLAM() {
  }

};

int main(int argc, char **argv) {
  ros::init(argc, argv, "image_converter");
  MY_SLAM slam;
  ros::spin();
  return 0;
}