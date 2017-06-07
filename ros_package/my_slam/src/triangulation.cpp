//
// Created by arqwer on 05.06.17.
//

#include "triangulation.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "my_geometry.h"
#include "my_util.h"

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

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

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


void FindFeatures(const Mat &frame1,
                  const Mat &frame2,
                  Mat &descriptors1,
                  Mat &descriptors2,
                  vector<KeyPoint> &keypoints1,
                  vector<KeyPoint> &keypoints2);

Mat NewExtrinsics(const Mat &prev_extrinsics, const Mat rotation, const Mat translation) {
  //todo:test
  Mat new_extrinsics = prev_extrinsics.clone();
  Mat rotation_reg = new_extrinsics(Rect(0, 0, 3, 3));
  Mat translation_reg = new_extrinsics(Rect(3, 0, 1, 3));
  Mat new_rotation = rotation * rotation_reg;
  Mat new_translation = translation_reg + translation;
  new_rotation.copyTo(rotation_reg);
  new_translation.copyTo(translation_reg);
  return new_extrinsics;
}

Mat NewExtrinsicMatrixFromEssential(const Mat &prev_extrinsic,
                                    const Mat &essential_mat,
                                    const Mat &cam_intrinsics,
                                    const vector<Point2f> &corrected_points_1,
                                    const vector<Point2f> &corrected_points_2) {
  //todo:test

  Mat rotation;
  Mat translation;
  recoverPose(essential_mat,
              corrected_points_1,
              corrected_points_2,
              cam_intrinsics,
              rotation,
              translation);
  return NewExtrinsics(prev_extrinsic, rotation, translation);
}

void DrawAll(const vector<KeyPoint> &keypoints1,
             const vector<KeyPoint> &keypoints2,
             int N_matches,
             const vector<Point2f> &corrected_points_1,
             const vector<Point2f> &corrected_points_2,
             Mat &frame1,
             Mat &frame2) {
  Size sz1 = frame1.size();
  Size sz2 = frame2.size();
  Mat display_matrix(sz1.height, sz1.width + sz2.width, CV_8UC3);
  Mat img1(display_matrix, Rect(0, 0, sz1.width, sz1.height));
  frame1.copyTo(img1);
  Mat img2(display_matrix, Rect(sz1.width, 0, sz2.width, sz2.height));
  frame2.copyTo(img2);
  drawKeypoints(img1, keypoints1, img1, Scalar({20, 240, 20}));
  drawKeypoints(img2, keypoints2, img2, Scalar({20, 240, 20}));

  vector<KeyPoint> keypoints1_cor;
  vector<KeyPoint> keypoints2_cor;
  for (size_t i = 0; i < corrected_points_1.size(); i++) {
    keypoints1_cor.push_back(KeyPoint(corrected_points_1[i], 1.f));
    keypoints2_cor.push_back(KeyPoint(corrected_points_2[i], 1.f));
  }
//    drawKeypoints(img1, keypoints1_cor, img1, Scalar({220, 20, 20}));
//    drawKeypoints(img2, keypoints2_cor, img2, Scalar({220, 20, 20}));

  for (int i_match = 0; i_match < N_matches; ++i_match) {
    line(display_matrix,
         keypoints1_cor[i_match].pt,
         keypoints2_cor[i_match].pt + Point2f(sz1.width, 0),
         hashcolor(i_match, 4));
  }
  for (int i_match = 0; i_match < N_matches; ++i_match) {
    circle(img1, keypoints1_cor[i_match].pt, 2, hashcolor(i_match, 2), 2);
    circle(img1, keypoints1_cor[i_match].pt, 3, hashcolor(i_match, 3));
    circle(img1, keypoints1_cor[i_match].pt, 4, hashcolor(i_match, 4));

    circle(img2, keypoints2_cor[i_match].pt, 2, hashcolor(i_match, 2), 2);
    circle(img2, keypoints2_cor[i_match].pt, 3, hashcolor(i_match, 3));
    circle(img2, keypoints2_cor[i_match].pt, 4, hashcolor(i_match, 4));
  }
  imshow("display_matrix", display_matrix);
}

void Triangulate2Frames(cv::Mat frame1,
                        cv::Mat frame2,
                        const Mat &camera_intrinsic,
                        vector<pcl::PointXYZRGB> *points,
                        vector<Mat> *out_points_covs,
                        Mat &out_descriptors) {
  assert(frame1.rows > 0);
  assert(frame1.cols > 0);
  assert(frame2.rows > 0);
  assert(frame2.cols > 0);
  //todo: set out_descriptors and out_pt_cov
  Mat descriptors1;
  Mat descriptors2;
  vector<KeyPoint> keypoints1;
  vector<KeyPoint> keypoints2;
  FindFeatures(frame1,
               frame2,
               descriptors1,
               descriptors2,
               keypoints1,
               keypoints2);

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

  //2.2 arrange descriptors
  out_descriptors = descriptors1.clone();
  for (int i = 0; i < queryIdxs.size(); ++i) {
    descriptors1.row(queryIdxs[i]).copyTo(out_descriptors.row(i));
  }
  out_descriptors.resize(points1.size());

  //3. Estimate Fundamental matrix
  Mat fundamental_mat = findFundamentalMat(points1, points2);
  Mat essential_mat = findEssentialMat(points1, points2, camera_intrinsic);

  //4. Correct 2d coordinates
  vector<Point2f> corrected_points_1;
  vector<Point2f> corrected_points_2;
  correctMatches(fundamental_mat, points1, points2, corrected_points_1, corrected_points_2);

  //5. Find projection matrices

//  Mat P1, P2;
//  projectionsFromFundamental(fundamental_mat, P1, P2);
  Mat extrinsic1(3, 4, CV_64F, Scalar(0));
  extrinsic1.at<double>(0, 0) = 1;
  extrinsic1.at<double>(1, 1) = 1;
  extrinsic1.at<double>(2, 2) = 1;
  Mat extrinsic2 = NewExtrinsicMatrixFromEssential(extrinsic1,
                                                   essential_mat,
                                                   camera_intrinsic,
                                                   corrected_points_1,
                                                   corrected_points_2);
  Mat P1 = camera_intrinsic * extrinsic1;
  Mat P2 = camera_intrinsic * extrinsic2;

  //6. Triangulate points and generate output
  points->resize(0);
  for (int i_match = 0; i_match < N_matches; ++i_match) {
    Mat pt_homog1(3, 1, CV_64FC1), pt_homog2(3, 1, CV_64FC1), pt3d_homog;
    double x1 = corrected_points_1[i_match].x;
    double x2 = corrected_points_2[i_match].x;
    double y1 = corrected_points_1[i_match].y;
    double y2 = corrected_points_2[i_match].y;
    pt_homog1.at<double>(0, 0) = x1;
    pt_homog1.at<double>(1, 0) = y1;
    pt_homog1.at<double>(2, 0) = 1;
    pt_homog2.at<double>(0, 0) = x2;
    pt_homog2.at<double>(1, 0) = y2;
    pt_homog2.at<double>(2, 0) = 1;
    pt3d_homog = triangulate_Linear_LS(P1, P2, pt_homog1, pt_homog2);
    Mat roi1(frame1, Rect(x1 - 1, y1 - 1, 3, 3));
    Mat roi2(frame2, Rect(x2 - 1, y2 - 1, 3, 3));
    Scalar pt_color = (mean(roi1) + mean(roi2)) / 2;
    double x, y, z, mul;
    mul = pt3d_homog.at<double>(3, 0);
    x = pt3d_homog.at<double>(0, 0) / mul;
    y = pt3d_homog.at<double>(1, 0) / mul;
    z = pt3d_homog.at<double>(2, 0) / mul;
    pcl::PointXYZRGB point3d(pt_color[2], pt_color[1], pt_color[0]);
    point3d.x = x;
    point3d.y = y;
    point3d.z = z;
    points->push_back(point3d);
  }

  //7. set covariances

  out_points_covs->resize(0);
  for (int i_pt = 0; i_pt < N_matches; ++i_pt) {
    Vec3d rel_ray = RayFromXY_rel(points1[i_pt].x, points1[i_pt].y, camera_intrinsic);
    int uncertainty_length = 200;// todo: calculate properly
    Mat sigma3d = CovarianceAlongLine(rel_ray[0],
                                      rel_ray[1],
                                      rel_ray[2],
                                      uncertainty_length, // todo: calculate properly
                                      0.02 * uncertainty_length);
    out_points_covs->push_back(sigma3d.clone());
  }

//  DrawAll(keypoints1, keypoints2, N_matches,corrected_points_1, corrected_points_2,frame1,frame2);
}

void FindFeatures(const Mat &frame1,
                  const Mat &frame2,
                  Mat &descriptors1,
                  Mat &descriptors2,
                  vector<KeyPoint> &keypoints1,
                  vector<KeyPoint> &keypoints2) {
//Default ORB parameters
  int nfeatures = 300;
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
  detector->detectAndCompute(frame1, noArray(), keypoints1, descriptors1);
  detector->detectAndCompute(frame2, noArray(), keypoints2, descriptors2);
}



void getTwoFrames(String filename, int frame_n1, int frame_n2, Mat *out1, Mat *out2) {
  VideoCapture cap(filename);
  cout << 123 << endl;
  if (!cap.isOpened()) {
    cout << "Can't open video" << endl;
    return;
  }
  for (int i_frame = 0; i_frame <= max(frame_n1, frame_n2); ++i_frame) {
    cout << i_frame << endl;
    Mat cur_frame;
    cap >> cur_frame;
    if (i_frame == frame_n1) {
      cur_frame.copyTo(*out1);
//      out1 = cur_frame.clone();
    }
    if (i_frame == frame_n2) {
      cur_frame.copyTo(*out2);
//      out2 = cur_frame.clone();
    }
  }
}

