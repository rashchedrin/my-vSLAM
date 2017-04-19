//
// Created by arqwer on 18.04.17.
//

#include "my_util.h"

#include <stdint.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

uint32_t uinthash(uint32_t x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = (x >> 16) ^ x;
  return x;
}

int32_t inthash(int32_t val, int32_t salt, int32_t low, int32_t high) {
  val = uinthash(val) + uinthash(uinthash(salt));
  val = val < 0 ? -val : val;
  return (val % (high - low)) + low;
}

Scalar hashcolor(int32_t val, int32_t salt) {
  if(salt == 0 ) {
    switch (val) {
      case 0: return Scalar(256, 0, 0);
        break;
      case 1: return Scalar(0, 256, 0);
        break;
      case 2: return Scalar(0, 0, 256);
        break;
      case 3: return Scalar(0, 0, 0);
        break;
    }
  }
  return Scalar({inthash(val, inthash(salt + 1)), inthash(val, inthash(salt + 2)),
                 inthash(val, inthash(salt + 3))});
}

void display_mat(Mat m, string name) {
  cout << name << " " << m.rows << " x " << m.cols << endl;
  for (int ir = 0; ir < m.rows; ++ir) {
    for (int ic = 0; ic < m.cols; ++ic) {
      cout << m.at<double>(ir, ic) << " ";
    }
    cout << endl;
  }
}


void DrawCross(Mat output_mat, Point2d pt, Scalar color, int size){
  Point2d s1 = Point2d(size, size);
  Point2d s2 = Point2d(size, -size);
  line(output_mat, pt - s1, pt + s1, color, 2);
  line(output_mat, pt - s2, pt + s2, color, 2);
  line(output_mat, pt - s1, pt + s1, Scalar(255,255,255), 1);
  line(output_mat, pt - s2, pt + s2, Scalar(255,255,255), 1);
}