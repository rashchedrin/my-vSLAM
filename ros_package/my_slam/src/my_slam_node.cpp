#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
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

  Mat known_descriptors_ORB;
  Mat known_descriptors_ORB_HD;
  Mat known_descriptors_SURF;
  Mat known_descriptors_SIFT;

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
    unsigned char known_descriptors_array_ORB[4][32] = {
        {188,  65,  60,  17, 171, 217, 125,  10, 238, 174,  80, 171, 253, 220,   3, 190, 230, 114, 176,   6, 108, 110, 247,  62,  72, 108, 181, 168, 100, 170, 203, 206},
        {172, 113, 232,  17, 171, 216, 178, 138,  46, 230, 144, 171, 124, 248, 169, 186, 186,  66, 112, 158, 200, 111, 242,  56,  66, 108, 245,  43,  54,  10, 239, 159},
        { 78,  81,  30,   9, 171,  80, 178,  72,   2,  44,  16, 171, 252, 248, 173, 138, 139, 114, 120, 190, 236,  47,  16,  46, 226, 158, 195,  44,  38, 138, 239, 159},
        { 44, 138,  28, 131, 189, 217,  76, 140, 136, 186,  24, 131, 207, 187,  23, 178,  23,  98,  48, 208, 192, 125,  67, 202,  84,  88, 205, 187,  68, 109, 139, 174}
    };
    unsigned char known_descriptors_array_ORB_HD[4][32] = {
        {161,  89, 113, 219, 108, 141, 239,  84,  76,   0, 113,  82, 181, 153, 176,  97, 206,  56,  18, 202, 227,  54,  90, 113,  37, 184,  34,  40,  89, 155,  97, 168},
        { 89,  73,   7, 165,  16, 198,  47, 164, 164, 192, 194,  64, 245,  23,  33, 226,  96, 229, 120, 238,  73, 153, 145,  45, 177, 162, 238,   0,  40,  50,  33,  40},
        {208, 216, 158, 250, 170, 204,  56, 134, 190,  27, 162,  49, 181, 211,  96,  91, 167,  51, 109, 144,  58,  77, 130, 113, 201, 226, 122,  75, 128,   1, 167,  11},
        {109, 242, 128,   3, 109,  31,  67, 156,  91, 251, 181,  76, 202,  64, 219, 190, 157, 204, 110, 126, 139, 116, 249, 178,  44,  28,   5, 116, 202, 191,  90,  53}
    };
    known_descriptors_ORB_HD = Mat(4, 32, CV_8UC1, &known_descriptors_array_ORB_HD);
    float known_descriptors_array_SURF[4][64] = {
        {-0.0043962286, 0.0017025528, 0.0044427644, 0.0017397819, 0.012043152, -0.0072439746, 0.018430376, 0.0090185702, -0.006758654, -2.150046e-05, 0.011013293, 0.0044661677, 0.00027224078, -0.00054577907, 0.0031854722, 0.0017236113, -0.039141476, 0.013745425, 0.039444353, 0.013948705, 0.20237887, 0.22762066, 0.28743717, 0.30388233, -0.13846363, 0.26065508, 0.21870884, 0.28167969, 0.0035527057, 0.0072235069, 0.007220794, 0.012590507, -0.041799542, 0.019875191, 0.042955179, 0.020146463, 0.15513235, -0.18235953, 0.32645339, 0.29832566, -0.1386005, -0.2812525, 0.24943772, 0.30383098, 0.0042250329, 0.0068993578, 0.011962755, 0.014195786, -0.0020601775, 0.00045552404, 0.0020666237, 0.0010278237, -0.028131191, 0.015425676, 0.029303303, 0.015425676, 0.0017448664, -0.0019349714, 0.024099655, 0.0097669708, 0.0015173206, -0.00046619805, 0.0020900087, 0.00072931329},
        {0.0011727142, -0.0006938561, 0.0016189738, 0.0011369428, -0.0034366108, 0.0010066149, 0.0040538143, 0.0031116565, -0.001474075, -0.00064844068, 0.0031030413, 0.0030623812, -0.0004419836, -0.00051723979, 0.00046632873, 0.00063142809, 0.004586156, -0.0034990702, 0.01505079, 0.016589995, 0.19109061, 0.15854885, 0.24701068, 0.22660246, -0.20173839, 0.16894464, 0.22427912, 0.22098625, -0.01462734, -0.0040448704, 0.016262352, 0.012008155, -0.0069434573, 0.013595417, 0.015499025, 0.014973385, 0.26879355, -0.022466794, 0.27008206, 0.37563384, -0.31153309, -0.095747396, 0.36807144, 0.33880755, -0.0027394714, 0.0048121205, 0.0056645758, 0.0055733295, 0.0026161717, -0.0049708043, 0.0032716703, 0.0050403955, 3.6474874e-05, -0.068730332, 0.0031825984, 0.069201782, -0.011842776, -0.032811705, 0.014770261, 0.035507329, 8.5151602e-05, 0.00045853999, 0.00026674438, 0.00052087259},
        {-0.0021128212, 0.0014749824, 0.0056767608, 0.0032609734, 0.0108164, -0.0046259332, 0.036566745, 0.016017675, -0.0014615795, -0.0011219676, 0.0027588934, 0.003482345, 0.00024316547, 0.0003960772, 0.00039368053, 0.00042085277, 0.03714684, -0.023993909, 0.037380029, 0.024520665, 0.20672411, 0.11718412, 0.41016278, 0.27113852, -0.16437621, 0.11932494, 0.22318013, 0.31618702, -0.0060539679, -0.0079648579, 0.01507629, 0.027155139, 0.0076987045, -0.0050627529, 0.0084020123, 0.0052997707, 0.26698646, -0.23467952, 0.26981819, 0.24889635, -0.088355564, -0.15244997, 0.32873315, 0.3116349, 0.0019223235, 0.017072456, 0.031304669, 0.043165371, 5.8956579e-05, 0.00011395878, 0.00017184207, 0.00028009305, 0.016293477, -0.0038068504, 0.016379215, 0.0084300935, 0.0062616449, -0.0097492291, 0.031131877, 0.03873245, 0.0030593667, -0.0021148208, 0.0039465195, 0.0024991105},
        {0.0017568803, -0.0054956586, 0.0019446614, 0.0054956586, 0.031523891, -0.069426216, 0.031523891, 0.069426216, 0.023324268, -0.036733877, 0.024901496, 0.036770727, 0.00086421164, -0.0034305891, 0.0016441309, 0.0035228555, -0.0018604191, 0.0066161579, 0.0088677024, 0.0069679408, 0.21724588, 0.21911907, 0.29980892, 0.30202079, -0.071480207, 0.0091994852, 0.25874072, 0.38448745, 0.019426346, -0.036778189, 0.019426346, 0.038079008, 0.0021050679, 0.00059788546, 0.005224369, 0.0026161377, 0.31254116, -0.12214217, 0.4075352, 0.21460818, -0.18558149, -0.0032836667, 0.2465158, 0.24636158, -0.0028705304, 0.017697915, 0.0093356315, 0.032152731, 0.00066026871, -0.00011235991, 0.0015350215, 0.00075430231, 0.016330834, 0.0085762581, 0.031980578, 0.0098906541, 0.0090869796, -0.0015840904, 0.020612502, 0.016380589, 0.00025965797, -0.0013753158, 0.0012407704, 0.0035980295}
    };
    known_descriptors_SURF = Mat(4, 64, CV_32FC1, &known_descriptors_array_SURF);
    float known_descriptors_array_SIFT[4][128] = {
        {16, 86, 10, 10, 81, 27, 0, 0, 121, 141, 4, 1, 1, 0, 0, 5, 44, 30, 4, 1, 8, 16, 12, 27, 0, 0, 0, 0, 7, 92, 36, 4, 51, 14, 2, 62, 141, 9, 0, 3, 141, 52, 1, 3, 4, 0, 0, 66, 111, 13, 0, 0, 0, 1, 10, 83, 0, 0, 0, 0, 0, 48, 23, 6, 38, 1, 0, 55, 65, 16, 33, 69, 141, 51, 0, 1, 1, 0, 8, 119, 119, 93, 0, 0, 0, 0, 1, 10, 4, 11, 0, 0, 0, 4, 2, 1, 3, 0, 1, 1, 6, 24, 98, 71, 41, 32, 16, 3, 2, 1, 13, 94, 43, 125, 9, 0, 0, 0, 0, 4, 7, 59, 5, 2, 0, 0, 0, 0},
        {7, 4, 0, 0, 33, 61, 5, 2, 32, 46, 11, 0, 4, 9, 3, 8, 48, 14, 11, 2, 0, 8, 55, 100, 1, 1, 2, 3, 16, 75, 87, 18, 0, 0, 0, 1, 48, 43, 9, 1, 79, 5, 0, 0, 19, 26, 44, 25, 123, 77, 17, 9, 3, 8, 38, 123, 19, 23, 24, 114, 123, 85, 50, 26, 17, 34, 12, 5, 6, 14, 30, 16, 24, 40, 32, 0, 1, 13, 123, 48, 64, 79, 57, 15, 7, 11, 111, 46, 3, 20, 38, 123, 119, 38, 6, 1, 8, 53, 26, 3, 0, 0, 0, 1, 4, 72, 112, 1, 0, 0, 6, 4, 0, 45, 123, 11, 3, 4, 11, 2, 0, 2, 62, 59, 37, 14, 1, 0},
        {7, 35, 19, 1, 0, 24, 51, 29, 5, 8, 7, 10, 34, 59, 33, 28, 0, 0, 0, 1, 18, 49, 77, 30, 0, 0, 0, 0, 1, 56, 109, 11, 6, 0, 0, 0, 10, 23, 93, 64, 17, 0, 0, 18, 110, 43, 41, 41, 114, 7, 0, 3, 18, 24, 103, 121, 12, 2, 1, 6, 67, 121, 121, 58, 36, 12, 24, 32, 30, 7, 8, 36, 35, 4, 6, 53, 121, 8, 3, 5, 121, 121, 45, 10, 11, 2, 4, 34, 24, 36, 85, 121, 113, 35, 8, 10, 9, 11, 66, 66, 7, 0, 0, 6, 15, 5, 24, 34, 14, 2, 12, 27, 21, 36, 47, 12, 7, 13, 11, 7, 15, 13, 68, 32, 5, 4, 5, 11},
        {1, 62, 51, 0, 0, 1, 16, 6, 10, 124, 90, 0, 0, 0, 15, 30, 7, 124, 124, 0, 0, 4, 11, 8, 0, 124, 124, 1, 0, 0, 0, 0, 22, 1, 0, 0, 0, 3, 22, 26, 124, 27, 3, 1, 3, 15, 79, 124, 14, 42, 31, 5, 60, 124, 90, 40, 1, 104, 79, 1, 18, 26, 5, 0, 45, 3, 0, 0, 0, 1, 2, 14, 124, 124, 25, 5, 4, 4, 4, 33, 20, 56, 47, 37, 83, 59, 15, 7, 2, 6, 2, 1, 18, 72, 73, 2, 40, 2, 0, 0, 0, 0, 0, 28, 66, 19, 3, 0, 0, 0, 0, 45, 48, 42, 12, 2, 1, 1, 3, 7, 15, 18, 8, 0, 0, 15, 34, 6}
    };
    known_descriptors_SIFT = Mat(4, 128, CV_32FC1, &known_descriptors_array_SIFT);
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

    int nfeatures = 1000;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 15; // Changed default (31);
    int firstLevel = 0;
    int WTA_K = 2;
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

    /*int minHessian = 400;
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(minHessian);*/
    Mat kp_descriptors;
    detector->detectAndCompute(cv_ptr->image,noArray(), key_points, kp_descriptors);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<vector<DMatch> > matches;
    matcher->knnMatch(known_descriptors_ORB_HD, kp_descriptors, matches, 1);

    cv::imshow("raw", cv_ptr->image);
//    cout<<"matcher ok"<<endl;
    drawKeypoints(cv_ptr->image, key_points, cv_ptr->image, Scalar({20, 240, 20}));
    for(int i_known = 0; i_known < 4; ++i_known){
      Scalar color;
      switch (i_known) {
        case 0: color = Scalar(256, 0, 0);
          break;
        case 1: color = Scalar(0, 256, 0);
          break;
        case 2: color = Scalar(0, 0, 256);
          break;
        case 3: color = Scalar(0, 0, 0);
          break;
      }
      int closest_id = matches[i_known][0].trainIdx;
      circle(cv_ptr->image, key_points[closest_id].pt, 1, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 2, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 3, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 4, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 5, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 6, color);
      circle(cv_ptr->image, key_points[closest_id].pt, 7, color);
    }

    /*
      for (int i_known_kp = 0; i_known_kp < 4; ++i_known_kp) {
        double min_norm =
            norm(known_descriptors.row(i_known_kp), kp_descriptors.row(0), NORM_HAMMING2);
        int closest_id = 0;
        for (int i_kp = 0; i_kp < key_points.size(); ++i_kp) {
          double distance =
              norm(known_descriptors.row(i_known_kp), kp_descriptors.row(i_kp), NORM_HAMMING2);
          if (distance < min_norm) {
            min_norm = distance;
            closest_id = i_kp;
          }
        }
        Scalar color;
        switch (i_known_kp) {
          case 0: color = Scalar(256, 0, 0);
            break;
          case 1: color = Scalar(0, 256, 0);
            break;
          case 2: color = Scalar(0, 0, 256);
            break;
          case 3: color = Scalar(0, 0, 0);
            break;
        }
        circle(cv_ptr->image, key_points[closest_id].pt, 1, color);
        circle(cv_ptr->image, key_points[closest_id].pt, 2, color);
        circle(cv_ptr->image, key_points[closest_id].pt, 3, color);
        circle(cv_ptr->image, key_points[closest_id].pt, 4, color);
        circle(cv_ptr->image, key_points[closest_id].pt, 5, color);
        circle(cv_ptr->image, key_points[closest_id].pt, 6, color);
        circle(cv_ptr->image, key_points[closest_id].pt, 7, color);
      }
    */
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
