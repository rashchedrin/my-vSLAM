#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

//#include <pcl_ros/point_cloud.h>
//#include <pcl/point_types.h>

using namespace cv;
using namespace std;

static const std::string OPENCV_WINDOW = "Image window 1";

uint32_t uinthash(uint32_t x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}

int32_t inthash(int32_t val, int32_t salt = 98262, int32_t low = 0, int32_t high = 256) {
  val = uinthash(val) + uinthash(uinthash(salt));
  val = val < 0 ? -val : val;
  return (val % (high - low)) + low;
}

Scalar hashcolor(int32_t val, int32_t salt = 35434) {
  return Scalar({inthash(val, inthash(salt + 1)), inthash(val, inthash(salt + 2)),
                 inthash(val, inthash(salt + 3))});
}

class Quaternion {
 public:
  double data[4];
  Quaternion() {}

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
  res[1] =  q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2] + q1[0] * q2[1];
  res[2] = -q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1] + q1[0] * q2[2];
  res[3] =  q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0] + q1[0] * q2[3];
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
};



class MY_SLAM {

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;

  image_transport::Subscriber image_sub_;
  ros::Publisher pub;

  StateMean x_state_mean;
  Mat P_state_cov;

  Mat known_descriptors;

  static const int nfeatures = 1000;
  //Default ORB parameters
  static const float scaleFactor = 1.2f;
  static const int nlevels = 8;
  static const int edgeThreshold = 15; // Changed default (31);
  Ptr<ORB> detector = ORB::create(
      nfeatures,
      scaleFactor,
      nlevels,
      edgeThreshold
  );

 public:

  MY_SLAM()
      : it_(nh_) {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("image_raw", 1,
                               &MY_SLAM::subscription_callback, this);
//    pub = nh_.advertise<sensor_msgs::PointCloud2>("output", 1);
//    PointCloud msg_out;
//    static uint64_t stamp = 0;
//    msg_out.header.frame_id = "map";
//    msg_out.height = msg_out.width = 100;
//    msg_out.header.stamp = ++stamp;


    P_state_cov = Mat::zeros(25, 25, CV_32F); //may be ones?
    x_state_mean.angular_velocity_r = Point3d(0,0,0);
    x_state_mean.direction_w = Quaternion(1,0,0,0);
    x_state_mean.position_w = Point3d(0,0,0);
    x_state_mean.velocity_w = Point3d(0,0,0);
    x_state_mean.feature_positions_w.push_back(Point3d (-40, 128.3, 4)); //cm
    x_state_mean.feature_positions_w.push_back(Point3d (-40, 128.3, 41));
    x_state_mean.feature_positions_w.push_back(Point3d (40, 128.3, 41));
    x_state_mean.feature_positions_w.push_back(Point3d (40, 128.3, 4));
    unsigned char known_descriptors_array[4][32] = {
        {188,  65,  60,  17, 171, 217, 125,  10, 238, 174,  80, 171, 253, 220,   3, 190, 230, 114, 176,   6, 108, 110, 247,  62,  72, 108, 181, 168, 100, 170, 203, 206},
        {172, 113, 232,  17, 171, 216, 178, 138,  46, 230, 144, 171, 124, 248, 169, 186, 186,  66, 112, 158, 200, 111, 242,  56,  66, 108, 245,  43,  54,  10, 239, 159},
        { 78,  81,  30,   9, 171,  80, 178,  72,   2,  44,  16, 171, 252, 248, 173, 138, 139, 114, 120, 190, 236,  47,  16,  46, 226, 158, 195,  44,  38, 138, 239, 159},
        { 44, 138,  28, 131, 189, 217,  76, 140, 136, 186,  24, 131, 207, 187,  23, 178,  23,  98,  48, 208, 192, 125,  67, 202,  84,  88, 205, 187,  68, 109, 139, 174}
    };
    known_descriptors = Mat(4, 32, CV_8UC1, &known_descriptors_array);
  }

  ~MY_SLAM() {
  }

  void subscription_callback(const sensor_msgs::ImageConstPtr &msg_in) {
    show_orb(msg_in);
  }

  void show_orb(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    std::vector<KeyPoint> key_points;

    //Default ORB parameters
    int nfeatures = 1000;
    float scaleFactor = 1.2f;
    int nlevels = 2;
    int edgeThreshold = 15; // Changed default (31);
    int firstLevel = 0;
    int WTA_K = 4;
    int scoreType = ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;

    Ptr<ORB> detector = ORB::create(
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
    Mat kp_descriptors;
    detector->detectAndCompute(cv_ptr->image,noArray(), key_points, kp_descriptors);

    cv::imshow("raw", cv_ptr->image);
    drawKeypoints(cv_ptr->image, key_points, cv_ptr->image, Scalar({20, 240, 20}));
    for(int  i_known_kp = 0; i_known_kp < 4; ++i_known_kp) {
      double min_norm = norm(known_descriptors.row(i_known_kp), kp_descriptors.row(0), NORM_HAMMING2);
      int closest_id = 0;
      for (int i_kp = 0; i_kp < key_points.size(); ++i_kp) {
        double distance = norm(known_descriptors.row(i_known_kp), kp_descriptors.row(i_kp), NORM_HAMMING2);
        if(distance < min_norm){
          min_norm = distance;
          closest_id = i_kp;
        }
      }
      Scalar color;
      switch(i_known_kp){
        case 0: color = Scalar(256,0,0); break;
        case 1: color = Scalar(0,256,0); break;
        case 2: color = Scalar(0,0,256); break;
        case 3: color = Scalar(0,0,0); break;
      }
      circle(cv_ptr->image, key_points[closest_id].pt, 1, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 2, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 3, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 4, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 5, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 6, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 7, color);
    }
    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey();

  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "image_converter");
  MY_SLAM slam;
  ros::spin();
  return 0;
}
