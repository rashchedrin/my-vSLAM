//
// Created by arqwer on 18.04.17.
//

#include "my_util.h"

#include <stdint.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

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