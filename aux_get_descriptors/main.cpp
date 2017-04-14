#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

#include<math.h>

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

double pifag(double a, double b){
  return sqrt(a*a + b*b);
}

int main() {
  VideoCapture stream1("/home/arqwer/Downloads/VID_HD.mp4");

  if (!stream1.isOpened()) { //check if video device has been initialised
    cout << "cannot open video stream";
  }


  int nfeatures = 100;
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
  Ptr<SURF> detector = SURF::create( minHessian );*/


  Mat frame;
  stream1.read(frame);

  Mat descriptors;
  std::vector<KeyPoint> keypoints;

  detector->detectAndCompute(frame, noArray(), keypoints, descriptors);
  cout<<type2str(descriptors.type())<<endl;
  cout<<descriptors.rows<<" x "<<descriptors.cols<<endl;
//  drawKeypoints(frame, keypoints, frame, Scalar({20, 0, 20}));
  int pt_coords[4][2] = {{289,348}, {295, 50}, {926,46}, {935,341}};
  for(int i_real = 0; i_real < 4; ++i_real) {
    int x_r = pt_coords[i_real][0];
    int y_r = pt_coords[i_real][1];

    int closest_kp = 0;
    double min_dist = pifag(keypoints[0].pt.x - x_r, keypoints[0].pt.y - y_r);
    for (int i_kp = 0; i_kp < keypoints.size(); ++i_kp) {
      double dist = pifag(keypoints[i_kp].pt.x - x_r, keypoints[i_kp].pt.y - y_r);
      if(dist < min_dist){
        closest_kp = i_kp;
        min_dist = dist;
      }
    }
//    cout<<i_real<<": "<<closest_kp<<endl;
    cout<<descriptors.row(closest_kp)<<endl;
    circle(frame, keypoints[closest_kp].pt, 3, Scalar({0,0,0}));
  }
//  cout<<descriptors<<endl;
  imshow("cam", frame);
  waitKey();
  return 0;
}