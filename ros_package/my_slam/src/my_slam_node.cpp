#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"

#include "my_util.h"
#include "my_types.h"
#include "jacobians.h"


//#include <pcl_ros/point_cloud.h>
//#include <pcl/point_types.h>

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
  result.direction_w =
      s0.direction_w * Quaternion(delta_time * (s0.angular_velocity_r + delta_angular_vel));
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

vector<Point2d> predict_points(const StateMean &s, const Mat &camIntrinsics) {
  vector<Point2d> result;
  for (int i_pt = 0; i_pt < s.feature_positions_w.size(); ++i_pt) {
    double q1 = s.direction_w.w();
    double q2 = s.direction_w.i();
    double q3 = s.direction_w.j();
    double q4 = s.direction_w.k();
    double r1 = s.position_w.x;
    double r2 = s.position_w.y;
    double r3 = s.position_w.z;
    double y1 = s.feature_positions_w[i_pt].x;
    double y2 = s.feature_positions_w[i_pt].y;
    double y3 = s.feature_positions_w[i_pt].z;
    double alpha_x = camIntrinsics.at<double>(0, 0);
    double alpha_y = camIntrinsics.at<double>(1, 1);
    double x0 = camIntrinsics.at<double>(0, 2);
    double y0 = camIntrinsics.at<double>(1, 2);

    double pred_x = x0 + (((pow(q3, 2) + pow(q4, 2)) * (r1 - y1) +
        pow(q1, 2) * (-r1 + y1) + pow(q2, 2) * (-r1 + y1) +
        2 * q1 * (q4 * (-r2 + y2) + q3 * (r3 - y3)) +
        2 * q2 * (-(q3 * r2) - q4 * r3 + q3 * y2 + q4 * y3)) * alpha_x) /
        (2 * q3 * q4 * r2 - pow(q3, 2) * r3 + pow(q4, 2) * r3 +
            2 * q2 * q4 * (r1 - y1) - 2 * q3 * q4 * y2 +
            2 * q1 * (q3 * (r1 - y1) + q2 * (-r2 + y2)) +
            pow(q1, 2) * (r3 - y3) + pow(q3, 2) * y3 -
            pow(q4, 2) * y3 + pow(q2, 2) * (-r3 + y3));
    double pred_y = y0 + ((pow(q3, 2) * r2 - pow(q4, 2) * r2 + 2 * q3 * q4 * r3 +
        2 * q1 * q4 * (-r1 + y1) + pow(q1, 2) * (r2 - y2) -
        pow(q3, 2) * y2 + pow(q4, 2) * y2 +
        pow(q2, 2) * (-r2 + y2) +
        2 * q2 * (q3 * (r1 - y1) + q1 * (r3 - y3)) - 2 * q3 * q4 * y3) *
        alpha_y) / (-2 * q3 * q4 * r2 + pow(q3, 2) * r3 -
        pow(q4, 2) * r3 + 2 * q2 * q4 * (-r1 + y1) +
        2 * q1 * (q3 * (-r1 + y1) + q2 * (r2 - y2)) + 2 * q3 * q4 * y2 +
        pow(q2, 2) * (r3 - y3) - pow(q3, 2) * y3 +
        pow(q4, 2) * y3 + pow(q1, 2) * (-r3 + y3));
    result.push_back(Point2d(pred_x, pred_y));
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
                       const StateMean &s,
                       double delta_time,
                       const Mat &Pn_noise_cov) {
  Mat result = Sigma_full.clone();
  Mat Sigma_cam_area = result(Rect(0, 0, 13, 13));
  Mat Sigma_cam_pred = predict_Sigma_cam(Sigma_cam_area, s, delta_time, Pn_noise_cov);
  Sigma_cam_pred.copyTo(Sigma_cam_area);
  return result;
}

//3N+13 x 2N
Mat Kalman_Gain(const Mat &H_t, double Q_sensor_noise,
                const StateMean &state,
                const Mat &Pn_noise_cov, const Mat &Sigma_predicted) {
  Mat innovation_cov = S_t_innovation_cov(H_t, Sigma_predicted, Q_sensor_noise);
  return Sigma_predicted * H_t.t() * innovation_cov.inv();
}

class MY_SLAM {

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;

  image_transport::Subscriber image_sub_;
  ros::Publisher pub;

  StateMean x_state_mean;
  Mat Sigma_state_cov;
  Mat Pn_noise_cov; // 6 x 6

//<editor-fold desc="descriptors">
  unsigned char known_descriptors_array_ORB[4][32] = {
      {188, 65, 60, 17, 171, 217, 125, 10, 238, 174, 80, 171, 253, 220, 3, 190, 230, 114, 176, 6,
       108, 110, 247, 62, 72, 108, 181, 168, 100, 170, 203, 206},
      {172, 113, 232, 17, 171, 216, 178, 138, 46, 230, 144, 171, 124, 248, 169, 186, 186, 66, 112,
       158, 200, 111, 242, 56, 66, 108, 245, 43, 54, 10, 239, 159},
      {78, 81, 30, 9, 171, 80, 178, 72, 2, 44, 16, 171, 252, 248, 173, 138, 139, 114, 120, 190,
       236, 47, 16, 46, 226, 158, 195, 44, 38, 138, 239, 159},
      {44, 138, 28, 131, 189, 217, 76, 140, 136, 186, 24, 131, 207, 187, 23, 178, 23, 98, 48, 208,
       192, 125, 67, 202, 84, 88, 205, 187, 68, 109, 139, 174}
  };
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
  const Mat known_descriptors_ORB_HD = Mat(4, 32, CV_8UC1, &known_descriptors_array_ORB_HD);

//</editor-fold>
  //calibrated
  double camera_intrinsic_array[3][3] = {{-1.0166722592048205e+03, 0., 6.3359083662958744e+02},
                                         {0., -1.0166722592048205e+03, 3.5881262886802512e+02},
                                         {0., 0., 1.}};
  //artificial, with principle point in center
  double camera_intrinsic_array2[3][3] = {{-1.0166722592048205e+03, 0., 612.105},
                                          {0., -1.0166722592048205e+03, 375.785},
                                          {0., 0., 1.}};
  const Mat camera_intrinsic = Mat(3, 3, CV_64F, &camera_intrinsic_array);

  Ptr<ORB> ORB_detector;
 public:

  constexpr static const double pixel_noise = 5.5;
  constexpr static const double position_speed_noise  = 1;
  constexpr static const double angular_speed_noise = 1;
  constexpr static const double initial_map_uncertainty = 0;
  constexpr static const double initial_pos_uncertainty = 100;
  constexpr static const double initial_direction_uncertainty = 10;
  MY_SLAM()
      : it_(nh_) {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("image_raw", 1,
                               &MY_SLAM::subscription_callback, this);
    Sigma_state_cov = initial_map_uncertainty*Mat::eye(25, 25, CV_64F);
    Sigma_state_cov.at<double>(0,0) = initial_pos_uncertainty;
    Sigma_state_cov.at<double>(1,1) = initial_pos_uncertainty;
    Sigma_state_cov.at<double>(2,2) = initial_pos_uncertainty;
    Sigma_state_cov.at<double>(3,3) = initial_direction_uncertainty;
    Sigma_state_cov.at<double>(4,4) = initial_direction_uncertainty;
    Sigma_state_cov.at<double>(5,5) = initial_direction_uncertainty;
    Sigma_state_cov.at<double>(6,6) = initial_direction_uncertainty;
    Pn_noise_cov = position_speed_noise*Mat::eye(6, 6, CV_64F); //todo: init properly
    Pn_noise_cov.at<double>(3,3) = angular_speed_noise;
    Pn_noise_cov.at<double>(4,4) = angular_speed_noise;
    Pn_noise_cov.at<double>(5,5) = angular_speed_noise;
    //Todo: fix no-rotation == 2Pi. Really it only works if delta_time = 1
    x_state_mean.angular_velocity_r =
        Point3d(2 * pi, 0, 0);// no rotation, 2*pi needed to eliminate indeterminance
    x_state_mean.direction_w = Quaternion(1, 0,  0, 0.0); // Zero rotation
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

  void subscription_callback(const sensor_msgs::ImageConstPtr &msg_in) {
    static int frame_number = 0;
    static int call_counter = 0;
    int msg_number = std::stoi(msg_in->header.frame_id);
    frame_number++;
    call_counter++;
    if (frame_number != msg_number) {
      cout << "FRAMES MISSED" << endl;
      frame_number = msg_number;
    }

    Mat image_in = ImageFromMsg(msg_in);
    cv::imshow("raw", image_in);
    EKF_iteration(image_in);
    cout<<"Frame: "<<frame_number<<" call: "<<call_counter<<endl;

    if (isnan(x_state_mean.position_w.x) ||
        isnan(x_state_mean.position_w.y) ||
        isnan(x_state_mean.position_w.z)) {
      cout << "Lost location" << endl;
      exit(0);
    }

    if (abs(norm(x_state_mean.direction_w) - 1) > 0.5) {
      cout << "Lost direction" << endl;
      cout << x_state_mean.direction_w << endl;
      cout << norm(x_state_mean.direction_w) << endl;
      exit(0);
    }

    if (norm(x_state_mean.position_w) > 500) {
      cout << "Position is too far" << endl;
      cout << x_state_mean.position_w << endl;
      cout << norm(x_state_mean.position_w) << endl;
      exit(0);
    }

    if (cv::waitKey(1) == 27) {
      exit(0);
    }
  }

  void EKF_iteration(Mat input_image) {
    cout << "state:" << endl << x_state_mean << endl;
    Mat output_mat = input_image.clone();
    double delta_time = 1; //todo: estimate properly. Warning! if delta_time is not 1, then angular_speed ==2pi will not be a 0 rotation
//Predict
    StateMean x_state_mean_pred =
        predict_state(x_state_mean, Point3d(0, 0, 0), Point3d(0, 0, 0), delta_time);
    cout << "st_pred:" << endl << x_state_mean_pred << endl;
    vector<Point2d> predicted_points = predict_points(x_state_mean_pred, camera_intrinsic);
    //yes, x_s_m
    Mat Sigma_state_cov_pred =
        predict_Sigma_full(Sigma_state_cov, x_state_mean, delta_time, Pn_noise_cov);
//Measure
    //Todo: make like in MonoSLAM, instead of ORB
    std::vector<KeyPoint> key_points;
    Mat kp_descriptors;
    ORB_detector->detectAndCompute(input_image, noArray(), key_points, kp_descriptors);

    auto &known_descriptors = known_descriptors_ORB_HD;

    vector<Point2d> observations = GetMatchingPointsCoordinates(key_points,
                                                                kp_descriptors,
                                                                known_descriptors);
    DrawPoints(output_mat, observations);
    DrawPoints(output_mat, predicted_points, 'x');
    Mat observations_diff =
        Mat(x_state_mean_pred.feature_positions_w.size() * 2, 1, CV_64F, double(0));
    for (int i_obs = 0; i_obs < x_state_mean_pred.feature_positions_w.size(); ++i_obs) {
      observations_diff.at<double>(i_obs * 2, 0) =
          observations[i_obs].x - predicted_points[i_obs].x;
      observations_diff.at<double>(i_obs * 2 + 1, 0) =
          observations[i_obs].y - predicted_points[i_obs].y;
    }

//Update
    //yes, x_pred
    Mat H_t = H_t_Jacobian_of_observations(x_state_mean_pred, camera_intrinsic);
    Mat KalmanGain =
        Kalman_Gain(H_t, pixel_noise, x_state_mean_pred, Pn_noise_cov, Sigma_state_cov_pred);
    Mat stateMat_pred = state2mat(x_state_mean_pred);
    StateMean x_state_mean_new = StateMean(stateMat_pred + KalmanGain * observations_diff);

    // this vv is mathematically incorrect, since EKF doesn't know about this line
    x_state_mean_new.direction_w =
        x_state_mean_new.direction_w / norm(x_state_mean_new.direction_w);

    Mat Sigma_state_cov_new = (
        Mat::eye(Sigma_state_cov.rows, Sigma_state_cov.cols, CV_64F) - KalmanGain * H_t)
        * Sigma_state_cov_pred;

    x_state_mean = x_state_mean_new;
    Sigma_state_cov = Sigma_state_cov_new;
    cout << endl;
    cv::imshow(OPENCV_WINDOW, output_mat);
  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "image_converter");
  MY_SLAM slam;
  ros::spin();
  return 0;
}