#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  VideoCapture cap;
  if (!cap.open(0)) {
    cerr << "Can't open camera" << endl;
    return 1;
  }
  cv::Mat frame;
  cap >> frame;
  for (;;) {
    cap >> frame;
    if (frame.empty()) {
      break;
    }// end of video stream
    if (waitKey(10) == 27) {
      break;
    } // stop capturing by pressing ESC

    std::vector<KeyPoint> key_points;

    // Default parameters of ORB
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

    detector->detect(frame, key_points);
    std::cout << "Found " << key_points.size() << " Keypoints " << std::endl;

    Mat out;
    drawKeypoints(frame, key_points, out, Scalar({20, 240, 20}));
    imshow("Orb keypoints", out);
  }
  // the camera will be closed automatically upon exit
  // cap.close();
  return 0;
}