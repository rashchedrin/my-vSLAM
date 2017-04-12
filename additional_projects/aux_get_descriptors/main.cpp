#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

#include<math.h>

double pifag(double a, double b){
  return sqrt(a*a + b*b);
}

int main() {
  VideoCapture stream1("../vid1.mp4");

  if (!stream1.isOpened()) { //check if video device has been initialised
    cout << "cannot open video stream";
  }


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

//unconditional loop
  Mat frame;
  stream1.read(frame);

  Mat descriptors;
  std::vector<KeyPoint> keypoints;

  detector->detectAndCompute(frame, noArray(), keypoints, descriptors);
  cout<<descriptors.rows<<" x "<<descriptors.cols<<endl;
//  drawKeypoints(frame, keypoints, frame, Scalar({20, 0, 20}));
  int pt_coords[4][2] = {{102, 236}, {103, 34}, {525, 29}, {534,228}};
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