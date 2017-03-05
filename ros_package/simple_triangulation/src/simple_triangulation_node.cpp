#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

using namespace cv;
using namespace std;

// Imported from opencv2/sfm
// https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/
template<typename T>
Mat
skewMat(const Mat_<T> &x) {
  Mat_<T> skew(3, 3);
  skew << 0, -x(2), x(1),
      x(2), 0, -x(0),
      -x(1), x(0), 0;

  return skew;
}

Mat
skew(InputArray _x) {
  const Mat x = _x.getMat();
  const int depth = x.depth();
  CV_Assert(x.size() == Size(3, 1) || x.size() == Size(1, 3));
  CV_Assert(depth == CV_32F || depth == CV_64F);

  Mat skewMatrix;
  if (depth == CV_32F) {
    skewMatrix = skewMat<float>(x);
  } else if (depth == CV_64F) {
    skewMatrix = skewMat<double>(x);
  } else {
    //CV_Error(CV_StsBadArg, "The DataType must be CV_32F or CV_64F");
  }

  return skewMatrix;
}

template<typename T>
void
projectionsFromFundamental(const Mat_<T> &F,
                           Mat_<T> P1,
                           Mat_<T> P2) {
  P1 << 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0;

  Vec<T, 3> e2;
  cv::SVD::solveZ(F.t(), e2);

  Mat_<T> P2cols = skew(e2) * F;
  for (char j = 0; j < 3; ++j) {
    for (char i = 0; i < 3; ++i)
      P2(j, i) = P2cols(j, i);
    P2(j, 3) = e2(j);
  }

}

void
projectionsFromFundamental(InputArray _F,
                           OutputArray _P1,
                           OutputArray _P2) {
  const Mat F = _F.getMat();
  const int depth = F.depth();
  CV_Assert(F.cols == 3 && F.rows == 3 && (depth == CV_32F || depth == CV_64F));

  _P1.create(3, 4, depth);
  _P2.create(3, 4, depth);

  Mat P1 = _P1.getMat(), P2 = _P2.getMat();

  // type
  if (depth == CV_32F) {
    projectionsFromFundamental<float>(F, P1, P2);
  } else {
    projectionsFromFundamental<double>(F, P1, P2);
  }

}

static const std::string OPENCV_WINDOW = "Image window";

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

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

//source: https://github.com/kipr/opencv/blob/master/samples/cpp/descriptor_extractor_matcher.cpp
static void crossCheckMatching(Ptr<DescriptorMatcher> &descriptorMatcher,
                               const Mat &descriptors1, const Mat &descriptors2,
                               vector<DMatch> &filteredMatches12, int knn = 1) {
  filteredMatches12.clear();
  vector<vector<DMatch> > matches12, matches21;
  descriptorMatcher->knnMatch(descriptors1, descriptors2, matches12, knn);
  descriptorMatcher->knnMatch(descriptors2, descriptors1, matches21, knn);
  for (size_t m = 0; m < matches12.size(); m++) {
    bool findCrossCheck = false;
    for (size_t fk = 0; fk < matches12[m].size(); fk++) {
      DMatch forward = matches12[m][fk];

      for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++) {
        DMatch backward = matches21[forward.trainIdx][bk];
        if (backward.trainIdx == forward.queryIdx) {
          filteredMatches12.push_back(forward);
          findCrossCheck = true;
          break;
        }
      }
      if (findCrossCheck) break;
    }
  }
}

//source:https://stackoverflow.com/questions/16295551/how-to-correctly-use-cvtriangulatepoints
// "the input parameters are two 3x4 camera projection matrices and a corresponding left/right pixel pair (x,y,w)."
Mat triangulate_Linear_LS(Mat mat_P_l, Mat mat_P_r, Mat warped_back_l, Mat warped_back_r) {
  Mat A(4, 3, CV_64FC1), b(4, 1, CV_64FC1), X(3, 1, CV_64FC1), X_homogeneous(4, 1, CV_64FC1),
      W(1, 1, CV_64FC1);
  W.at<double>(0, 0) = 1.0;
  A.at<double>(0, 0) =
      (warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0)) * mat_P_l.at<double>(2, 0)
          - mat_P_l.at<double>(0, 0);
  A.at<double>(0, 1) =
      (warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0)) * mat_P_l.at<double>(2, 1)
          - mat_P_l.at<double>(0, 1);
  A.at<double>(0, 2) =
      (warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0)) * mat_P_l.at<double>(2, 2)
          - mat_P_l.at<double>(0, 2);
  A.at<double>(1, 0) =
      (warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0)) * mat_P_l.at<double>(2, 0)
          - mat_P_l.at<double>(1, 0);
  A.at<double>(1, 1) =
      (warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0)) * mat_P_l.at<double>(2, 1)
          - mat_P_l.at<double>(1, 1);
  A.at<double>(1, 2) =
      (warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0)) * mat_P_l.at<double>(2, 2)
          - mat_P_l.at<double>(1, 2);
  A.at<double>(2, 0) =
      (warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0)) * mat_P_r.at<double>(2, 0)
          - mat_P_r.at<double>(0, 0);
  A.at<double>(2, 1) =
      (warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0)) * mat_P_r.at<double>(2, 1)
          - mat_P_r.at<double>(0, 1);
  A.at<double>(2, 2) =
      (warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0)) * mat_P_r.at<double>(2, 2)
          - mat_P_r.at<double>(0, 2);
  A.at<double>(3, 0) =
      (warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0)) * mat_P_r.at<double>(2, 0)
          - mat_P_r.at<double>(1, 0);
  A.at<double>(3, 1) =
      (warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0)) * mat_P_r.at<double>(2, 1)
          - mat_P_r.at<double>(1, 1);
  A.at<double>(3, 2) =
      (warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0)) * mat_P_r.at<double>(2, 2)
          - mat_P_r.at<double>(1, 2);
  b.at<double>(0, 0) =
      -((warped_back_l.at<double>(0, 0) / warped_back_l.at<double>(2, 0)) * mat_P_l.at<double>(2, 3)
          - mat_P_l.at<double>(0, 3));
  b.at<double>(1, 0) =
      -((warped_back_l.at<double>(1, 0) / warped_back_l.at<double>(2, 0)) * mat_P_l.at<double>(2, 3)
          - mat_P_l.at<double>(1, 3));
  b.at<double>(2, 0) =
      -((warped_back_r.at<double>(0, 0) / warped_back_r.at<double>(2, 0)) * mat_P_r.at<double>(2, 3)
          - mat_P_r.at<double>(0, 3));
  b.at<double>(3, 0) =
      -((warped_back_r.at<double>(1, 0) / warped_back_r.at<double>(2, 0)) * mat_P_r.at<double>(2, 3)
          - mat_P_r.at<double>(1, 3));
  solve(A, b, X, DECOMP_SVD);
  vconcat(X, W, X_homogeneous);
  return X_homogeneous;
}

///**
// From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
// */
//Mat_ LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
//                           Matx34d P,       //camera 1 matrix
//                           Point3d u1,      //homogenous image point in 2nd camera
//                           Matx34d P1       //camera 2 matrix
//)
//{
//  //build matrix A for homogenous equation system Ax = 0
//  //assume X = (x,y,z,1), for Linear-LS method
//  //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
//  Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
//            u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
//            u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
//            u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
//  );
//  Mat_ B = (Mat_(4,1) <<    -(u.x*P(2,3)    -P(0,3)),
//      -(u.y*P(2,3)  -P(1,3)),
//      -(u1.x*P1(2,3)    -P1(0,3)),
//      -(u1.y*P1(2,3)    -P1(1,3)));
//
//  Mat_ X;
//  solve(A,B,X,DECOMP_SVD);
//
//  return X;
//}

void TwoFrames2Cloud(cv::Mat frame1, cv::Mat frame2, PointCloud &points3D) {
  //1. Find features
  static bool first_time = true;
  int nfeatures = 1000;
  //Default ORB parameters
  float scaleFactor = 1.2f;
  int nlevels = 8;
  int edgeThreshold = 15; // Changed default (31);
  Ptr<ORB> detector = ORB::create(
      nfeatures,
      scaleFactor,
      nlevels,
      edgeThreshold
  );

  Mat descriptors1, descriptors2;
  std::vector<KeyPoint> keypoints1, keypoints2;
  detector->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
  detector->detectAndCompute(frame2, noArray(), keypoints2, descriptors2);

  //2. Find correspondance
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  vector<DMatch> filteredMatches;
  crossCheckMatching(matcher, descriptors1, descriptors2, filteredMatches, 1);
  //2.1 Convert filteredMatches to points
  vector<int> queryIdxs(filteredMatches.size()), trainIdxs(filteredMatches.size());
  for (size_t i = 0; i < filteredMatches.size(); i++) {
    queryIdxs[i] = filteredMatches[i].queryIdx;
    trainIdxs[i] = filteredMatches[i].trainIdx;
  }
  int N_matches = filteredMatches.size();
  vector<Point2f> points1;
  KeyPoint::convert(keypoints1, points1, queryIdxs);
  vector<Point2f> points2;
  KeyPoint::convert(keypoints2, points2, trainIdxs);

  //3. Estimate Fundamental matrix
  Mat F = findFundamentalMat(points1, points2);

  //4. Correct 2d coordinates
  vector<Point2f> corrected_points_1;
  vector<Point2f> corrected_points_2;
  correctMatches(F, points1, points2, corrected_points_1, corrected_points_2);
  //display matches and corrected keypoints
  {
    Size sz1 = frame1.size();
    Size sz2 = frame2.size();
    Mat display_matrix(sz1.height, sz1.width + sz2.width, CV_8UC3);
    Mat img1(display_matrix, Rect(0, 0, sz1.width, sz1.height));
    frame1.copyTo(img1);
    Mat img2(display_matrix, Rect(sz1.width, 0, sz2.width, sz2.height));
    frame2.copyTo(img2);
    drawKeypoints(img1, keypoints1, img1, Scalar({20, 240, 20}));
    drawKeypoints(img2, keypoints2, img2, Scalar({20, 240, 20}));

    std::vector<cv::KeyPoint> keypoints1_cor;
    std::vector<cv::KeyPoint> keypoints2_cor;
    for (size_t i = 0; i < corrected_points_1.size(); i++) {
      keypoints1_cor.push_back(cv::KeyPoint(corrected_points_1[i], 1.f));
      keypoints2_cor.push_back(cv::KeyPoint(corrected_points_2[i], 1.f));
    }
    drawKeypoints(img1, keypoints1_cor, img1, Scalar({220, 20, 20}));
    drawKeypoints(img2, keypoints2_cor, img2, Scalar({220, 20, 20}));

    for (int i_match = 0; i_match < N_matches; ++i_match) {
      line(display_matrix,
           keypoints1_cor[i_match].pt,
           keypoints2_cor[i_match].pt + Point2f(sz1.width, 0),
           hashcolor(i_match));
    }
    imshow("display_matrix", display_matrix);
  }

  //5. Find projection matrices
  Mat P1, P2;
  projectionsFromFundamental(F, P1, P2);

  //6. Triangulate points and generate output
  for (int i_match = 0; i_match < N_matches; ++i_match) {
    Mat pt_homog1(3, 1, CV_64FC1), pt_homog2(3, 1, CV_64FC1), pt3d_homog;
    pt_homog1.at<double>(0, 0) = corrected_points_1[i_match].x;
    pt_homog1.at<double>(1, 0) = corrected_points_1[i_match].y;
    pt_homog1.at<double>(2, 0) = 1;
    pt_homog2.at<double>(0, 0) = corrected_points_2[i_match].x;
    pt_homog2.at<double>(1, 0) = corrected_points_2[i_match].y;
    pt_homog2.at<double>(2, 0) = 1;
    pt3d_homog = triangulate_Linear_LS(P1, P2, pt_homog1, pt_homog2);
    double x, y, z, mul;
    mul = pt3d_homog.at<double>(3, 0);
    x = pt3d_homog.at<double>(0, 0) / mul;
    y = pt3d_homog.at<double>(1, 0) / mul;
    z = pt3d_homog.at<double>(2, 0) / mul;
    points3D.push_back(pcl::PointXYZ(x, y, z));
    if (first_time) {
      cout << x << "\t" << y << "\t" << z << endl;
    }
  }

//  Mat points4D(4, N_matches, CV_64FC1);
//  triangulatePoints(P1, P2, corrected_points_1, corrected_points_2, points4D);

//  for(int i_point = 0; i_point < N_matches; i_point++){
//    double x,y,z, mul;
//    mul = points4D.at<double>(3, i_point);
//    x = points4D.at<double>(0,i_point) / mul;
//    y = points4D.at<double>(1,i_point) / mul;
//    z = points4D.at<double>(2,i_point) / mul;
//    points3D.push_back(pcl::PointXYZ(x,y,z));
//    if(first_time){
//      cout<<x<<"\t"<<y<<"\t"<<z<<endl;/*
//      for(int i_coord = 0; i_coord < 4; ++i_coord){
//        cout<<points4D.at<double>(i_coord,i_point)<<"\t";
//      }
//      cout<<endl;*/
//    }
//  }

  first_time = false;
}

class OrbMarker {

  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;

  image_transport::Subscriber image_sub_;
  ros::Publisher pub;
 public:

  OrbMarker()
      : it_(nh_) {
    // Subscrive to input video feed and publish output video feed
    image_sub_ = it_.subscribe("image_raw", 1,
                               &OrbMarker::subscription_callback, this);
    pub = nh_.advertise<sensor_msgs::PointCloud2>("output", 1);
    cv::namedWindow(OPENCV_WINDOW);
    Mat frame1, frame2;
    frame1 = imread("boxa1.png", CV_LOAD_IMAGE_COLOR);
    frame2 = imread("boxa2.png", CV_LOAD_IMAGE_COLOR);

    if (!frame1.data || !frame2.data)                              // Check for invalid input
    {
      cout << "Could not open or find the image" << std::endl;
      return;
    }

    PointCloud msg_out;
    static uint64_t stamp = 0;
    msg_out.header.frame_id = "map";
    msg_out.height = msg_out.width = 100;
    msg_out.header.stamp = ++stamp;
    TwoFrames2Cloud(frame1, frame2, msg_out);
    cv::waitKey(100);
    pub.publish(msg_out);
  }

  ~OrbMarker() {
    cv::destroyWindow(OPENCV_WINDOW);
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

    cv::imshow("raw", cv_ptr->image);
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
