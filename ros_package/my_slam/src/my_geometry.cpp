//
// Created by arqwer on 23.04.17.
//

#include "my_types.h"
#include "my_geometry.h"

Vec3d RayFromXY_rel(int x, int y, const Mat &camIntrinsics) {
  double alpha_x = camIntrinsics.at<double>(0, 0);
  double alpha_y = camIntrinsics.at<double>(1, 1);
  double x0 = camIntrinsics.at<double>(0, 2);
  double y0 = camIntrinsics.at<double>(1, 2);
  double delta_x = x - x0;
  double delta_y = y - y0;
  double X_dir, Y_dir, Z_dir;
  Z_dir = 1;
  //Mind the sign
  X_dir = -delta_x / alpha_x * Z_dir;
  Y_dir = -delta_y / alpha_y * Z_dir;
  return Vec3d(X_dir, Y_dir, Z_dir);
}

Point2d ProjectPoint(Point3d pt_3d,
                     Point3d cam_position,
                     const Quaternion &rotation_wr,
                     const Mat &camIntrinsics) {
  double q1 = rotation_wr.w();
  double q2 = rotation_wr.i();
  double q3 = rotation_wr.j();
  double q4 = rotation_wr.k();
  double r1 = cam_position.x;
  double r2 = cam_position.y;
  double r3 = cam_position.z;
  double y1 = pt_3d.x;
  double y2 = pt_3d.y;
  double y3 = pt_3d.z;
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
  return Point2d(pred_x, pred_y);
}


Mat RotationMatXtoXYZ(double x, double y, double z) {
  if(y == 0 && z == 0){
    return Mat::eye(3,3,CV_64F);
  }
  double normV1 = sqrt(x * x + y * y + z * z);
  double normV2 = sqrt(y * y + z * z);
  double normV3 = normV1 * normV2;
  //@formatter:off
  double rot_array[3][3] =
      {
          {x / normV1,           0,  (-y * y - z * z) / normV3},
          {y / normV1,  z / normV2,           (x * y) / normV3},
          {z / normV1, -y / normV2,           (x * z) / normV3}
      };
  //@formatter:on
  Mat rot(3, 3, CV_64F, &rot_array);
  return rot.clone();
}



