#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

static const std::string OPENCV_WINDOW = "Image window";

class OrbMarker {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;

 public:
  OrbMarker()
      : it_(nh_) {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("image_raw", 1,
                               &OrbMarker::show_orb, this);
    cv::namedWindow(OPENCV_WINDOW);
  }

  ~OrbMarker() {
    cv::destroyWindow(OPENCV_WINDOW);
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
        fastThreshold);

    detector->detect(cv_ptr->image, key_points);

    drawKeypoints(cv_ptr->image, key_points, cv_ptr->image, Scalar({20, 240, 20}));

    // Update GUI Window
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    cv::waitKey(3);

  }
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "image_converter");
  OrbMarker ic;
  ros::spin();
  return 0;
}
