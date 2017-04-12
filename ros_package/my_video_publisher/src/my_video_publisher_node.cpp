#include<iostream>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace cv;

int main(int argc, char ** argv){
  Mat frame;
  VideoCapture cap("/home/arqwer/vid1_mp4");
  ros::init(argc, argv, "my_video_publisher");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Publisher pub = it.advertise("my_video_publisher/image", 1);


  ros::Rate loop_rate(25);
  while (nh.ok()) {
    cap >> frame;
    if(frame.empty()){
      break;
    }
//    imshow("123",frame);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}