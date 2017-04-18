//
// Created by arqwer on 18.04.17.
//

#ifndef MY_SLAM_MY_UTIL_H
#define MY_SLAM_MY_UTIL_H

#include <stdint.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

uint32_t uinthash(uint32_t x);

int32_t inthash(int32_t val, int32_t salt = 98262, int32_t low = 0, int32_t high = 256);

Scalar hashcolor(int32_t val, int32_t salt = 35434);

void display_mat(Mat m) {
  cout<<m.rows<<" x "<<m.cols<<endl;
  for (int ir = 0; ir < m.rows; ++ir) {
    for (int ic = 0; ic < m.cols; ++ic) {
      cout << m.at<double>(ir, ic) << " ";
    }
    cout << endl;
  }
}
#endif //MY_SLAM_MY_UTIL_H
