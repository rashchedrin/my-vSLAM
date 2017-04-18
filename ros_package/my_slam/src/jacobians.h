//
// Created by arqwer on 18.04.17.
//

#ifndef MY_SLAM_JACOBIANS_H
#define MY_SLAM_JACOBIANS_H

#include "my_types.h"

#include <stdint.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//13 x 6
Mat Q_df_over_dn(const StateMean &state, double dt);
Mat Q_df_over_dn(const StateMean &state, double dt) {
  double w1 = state.angular_velocity_r.x;
  double w2 = state.angular_velocity_r.y;
  double w3 = state.angular_velocity_r.z;
  double q1 = state.direction_w.w();
  double q2 = state.direction_w.i();
  double q3 = state.direction_w.j();
  double q4 = state.direction_w.k();

  double norm_w = pow(pow(w1, 2) + pow(w2, 2) + pow(w3, 2), 0.5);
  double norm_w_squared = pow(w1, 2) + pow(w2, 2) + pow(w3, 2);

  double result_array[13][6] = {
      {dt, 0, 0, 0, 0, 0},
      {0, dt, 0, 0, 0, 0},
      {0, dt, 0, 0, 0, 0},
      {0, 0, 0, (dt * (-(q3 * w1 * w2 * cos(norm_w / 2.) * pow(norm_w_squared, -1)) -
          q4 * w1 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) -
          q2 * cos(norm_w / 2.) * pow(w1, 2) * pow(norm_w_squared, -1) +
          2 * q3 * w1 * w2 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
          2 * q4 * w1 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
          2 * q2 * pow(w1, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
          2 * q2 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
          q1 * w1 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.,
       (dt * (-(q2 * w1 * w2 * cos(norm_w / 2.) * pow(norm_w_squared, -1)) -
           q4 * w2 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) -
           q3 * cos(norm_w / 2.) * pow(w2, 2) * pow(norm_w_squared, -1) +
           2 * q2 * w1 * w2 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q4 * w2 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q3 * pow(w2, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q3 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
           q1 * w2 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.,
       (dt * (-(q2 * w1 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1)) -
           q3 * w2 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) -
           q4 * cos(norm_w / 2.) * pow(w3, 2) * pow(norm_w_squared, -1) +
           2 * q2 * w1 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q3 * w2 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q4 * pow(w3, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q4 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
           q1 * w3 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.},
      {0, 0, 0, (dt * (-(q4 * w1 * w2 * cos(norm_w / 2.) * pow(norm_w_squared, -1)) +
          q3 * w1 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) +
          q1 * cos(norm_w / 2.) * pow(w1, 2) * pow(norm_w_squared, -1) +
          2 * q4 * w1 * w2 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
          2 * q3 * w1 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
          2 * q1 * pow(w1, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
          2 * q1 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
          q2 * w1 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.,
       (dt * (q1 * w1 * w2 * cos(norm_w / 2.) * pow(norm_w_squared, -1) +
           q3 * w2 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) -
           q4 * cos(norm_w / 2.) * pow(w2, 2) * pow(norm_w_squared, -1) -
           2 * q1 * w1 * w2 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q3 * w2 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q4 * pow(w2, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q4 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
           q2 * w2 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.,
       (dt * (q1 * w1 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) -
           q4 * w2 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) +
           q3 * cos(norm_w / 2.) * pow(w3, 2) * pow(norm_w_squared, -1) -
           2 * q1 * w1 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q4 * w2 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q3 * pow(w3, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q3 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
           q2 * w3 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.},
      {0, 0, 0, (dt * (q1 * w1 * w2 * cos(norm_w / 2.) * pow(norm_w_squared, -1) -
          q2 * w1 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) +
          q4 * cos(norm_w / 2.) * pow(w1, 2) * pow(norm_w_squared, -1) -
          2 * q1 * w1 * w2 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
          2 * q2 * w1 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
          2 * q4 * pow(w1, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
          2 * q4 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
          q3 * w1 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.,
       (dt * (q4 * w1 * w2 * cos(norm_w / 2.) * pow(norm_w_squared, -1) -
           q2 * w2 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) +
           q1 * cos(norm_w / 2.) * pow(w2, 2) * pow(norm_w_squared, -1) -
           2 * q4 * w1 * w2 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q2 * w2 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q1 * pow(w2, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q1 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
           q3 * w2 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.,
       (dt * (q4 * w1 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) +
           q1 * w2 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) -
           q2 * cos(norm_w / 2.) * pow(w3, 2) * pow(norm_w_squared, -1) -
           2 * q4 * w1 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q1 * w2 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q2 * pow(w3, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q2 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
           q3 * w3 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.},
      {0, 0, 0, (dt * (q2 * w1 * w2 * cos(norm_w / 2.) * pow(norm_w_squared, -1) +
          q1 * w1 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) -
          q3 * cos(norm_w / 2.) * pow(w1, 2) * pow(norm_w_squared, -1) -
          2 * q2 * w1 * w2 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
          2 * q1 * w1 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
          2 * q3 * pow(w1, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
          2 * q3 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
          q4 * w1 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.,
       (dt * (-(q3 * w1 * w2 * cos(norm_w / 2.) * pow(norm_w_squared, -1)) +
           q1 * w2 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) +
           q2 * cos(norm_w / 2.) * pow(w2, 2) * pow(norm_w_squared, -1) +
           2 * q3 * w1 * w2 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q1 * w2 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q2 * pow(w2, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q2 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
           q4 * w2 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.,
       (dt * (-(q3 * w1 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1)) +
           q2 * w2 * w3 * cos(norm_w / 2.) * pow(norm_w_squared, -1) +
           q1 * cos(norm_w / 2.) * pow(w3, 2) * pow(norm_w_squared, -1) +
           2 * q3 * w1 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q2 * w2 * w3 * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) -
           2 * q1 * pow(w3, 2) * pow(norm_w_squared, -1.5) * sin(norm_w / 2.) +
           2 * q1 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.) -
           q4 * w3 * pow(norm_w_squared, -0.5) * sin(norm_w / 2.))) / 2.},
      {1, 0, 0, 0, 0, 0},
      {0, 1, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0},
      {0, 0, 0, 1, 0, 0},
      {0, 0, 0, 0, 1, 0},
      {0, 0, 0, 0, 0, 1}};
  //todo: think about removing .clone()
  Mat result = Mat(13, 6, CV_64F, &result_array).clone();
  return result;
}

//13 x 13
Mat Ft_df_over_dxcam(const StateMean &state, double dt) {
  double norm_w = norm(state.angular_velocity_r);
  double result_array[13][13] = {
      {1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0, 0},
      {0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0, 0, 0, 0, dt, 0, 0, 0},
      {0, 0, 0, dt * cos(norm_w / 2.), -((dt * state.angular_velocity_r.x * sin(norm_w / 2.)) /
          norm_w), -((dt * state.angular_velocity_r.y * sin(norm_w / 2.)) / norm_w),
       -((dt * state.angular_velocity_r.z * sin(norm_w / 2.)) / norm_w), 0, 0, 0,
       -(dt * (state.angular_velocity_r.x * (state.direction_w.i() * state.angular_velocity_r.x
           + state.direction_w.j() * state.angular_velocity_r.y
           + state.direction_w.k() * state.angular_velocity_r.z) * norm_w * cos(norm_w / 2.) +
           (state.direction_w.w() * state.angular_velocity_r.x
               * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                   + pow(state.angular_velocity_r.z, 2)) + 2
               * (-(state.direction_w.j() * state.angular_velocity_r.x * state.angular_velocity_r.y)
                   - state.direction_w.k() * state.angular_velocity_r.x * state.angular_velocity_r.z
                   + state.direction_w.i()
                       * (pow(state.angular_velocity_r.y, 2) + pow(state.angular_velocity_r.z, 2))))
               * sin(norm_w / 2.))) /
           (2. * pow(pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                         + pow(state.angular_velocity_r.z, 2), 1.5)), -(dt
          * (state.angular_velocity_r.y * (state.direction_w.i() * state.angular_velocity_r.x
              + state.direction_w.j() * state.angular_velocity_r.y
              + state.direction_w.k() * state.angular_velocity_r.z) * norm_w *
              cos(norm_w / 2.) + (2 * state.direction_w.j()
              * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.z, 2))
              + state.angular_velocity_r.y
                  * (-2 * state.direction_w.i() * state.angular_velocity_r.x
                      - 2 * state.direction_w.k() * state.angular_velocity_r.z
                      + state.direction_w.w()
                          * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                              + pow(state.angular_velocity_r.z, 2)))) *
              sin(norm_w / 2.))) / (2. * pow(
          pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
              + pow(state.angular_velocity_r.z, 2), 1.5)),
       -(dt * (state.angular_velocity_r.z * (state.direction_w.i() * state.angular_velocity_r.x
           + state.direction_w.j() * state.angular_velocity_r.y
           + state.direction_w.k() * state.angular_velocity_r.z) * norm_w * cos(norm_w / 2.) +
           (2 * state.direction_w.k()
               * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2))
               + state.angular_velocity_r.z
                   * (-2 * state.direction_w.i() * state.angular_velocity_r.x
                       - 2 * state.direction_w.j() * state.angular_velocity_r.y
                       + state.direction_w.w() * (pow(state.angular_velocity_r.x, 2)
                           + pow(state.angular_velocity_r.y, 2)
                           + pow(state.angular_velocity_r.z, 2)))) * sin(norm_w / 2.))) /
           (2. * pow(pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                         + pow(state.angular_velocity_r.z, 2), 1.5))},
      {0, 0, 0,
       (dt * state.angular_velocity_r.x * sin(norm_w / 2.)) / norm_w, dt * cos(norm_w / 2.),
       (dt * state.angular_velocity_r.z * sin(norm_w / 2.)) / norm_w,
       -((dt * state.angular_velocity_r.y * sin(norm_w / 2.)) / norm_w), 0, 0, 0,
       (dt * (state.angular_velocity_r.x * (state.direction_w.w() * state.angular_velocity_r.x
           - state.direction_w.k() * state.angular_velocity_r.y
           + state.direction_w.j() * state.angular_velocity_r.z) * norm_w * cos(norm_w / 2.) -
           (state.direction_w.i() * state.angular_velocity_r.x
               * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                   + pow(state.angular_velocity_r.z, 2)) - 2
               * (state.direction_w.k() * state.angular_velocity_r.x * state.angular_velocity_r.y
                   - state.direction_w.j() * state.angular_velocity_r.x * state.angular_velocity_r.z
                   + state.direction_w.w()
                       * (pow(state.angular_velocity_r.y, 2) + pow(state.angular_velocity_r.z, 2))))
               * sin(norm_w / 2.))) /
           (2. * pow(pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                         + pow(state.angular_velocity_r.z, 2), 1.5)), (dt
          * (state.angular_velocity_r.y * (state.direction_w.w() * state.angular_velocity_r.x
              - state.direction_w.k() * state.angular_velocity_r.y
              + state.direction_w.j() * state.angular_velocity_r.z) * norm_w *
              cos(norm_w / 2.) - (2 * state.direction_w.k()
              * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.z, 2))
              + state.angular_velocity_r.y * (2 * state.direction_w.w() * state.angular_velocity_r.x
                  + 2 * state.direction_w.j() * state.angular_velocity_r.z + state.direction_w.i()
                  * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                      + pow(state.angular_velocity_r.z, 2)))) *
              sin(norm_w / 2.))) / (2. * pow(
          pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
              + pow(state.angular_velocity_r.z, 2), 1.5)),
       (dt * (state.angular_velocity_r.z * (state.direction_w.w() * state.angular_velocity_r.x
           - state.direction_w.k() * state.angular_velocity_r.y
           + state.direction_w.j() * state.angular_velocity_r.z) * norm_w * cos(norm_w / 2.) +
           (2 * state.direction_w.j()
               * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2))
               - state.angular_velocity_r.z
                   * (2 * state.direction_w.w() * state.angular_velocity_r.x
                       - 2 * state.direction_w.k() * state.angular_velocity_r.y
                       + state.direction_w.i() * (pow(state.angular_velocity_r.x, 2)
                           + pow(state.angular_velocity_r.y, 2)
                           + pow(state.angular_velocity_r.z, 2)))) * sin(norm_w / 2.))) /
           (2. * pow(pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                         + pow(state.angular_velocity_r.z, 2), 1.5))},
      {0, 0, 0,
       (dt * state.angular_velocity_r.y * sin(norm_w / 2.)) / norm_w,
       -((dt * state.angular_velocity_r.z * sin(norm_w / 2.)) / norm_w), dt * cos(norm_w / 2.),
       (dt * state.angular_velocity_r.x * sin(norm_w / 2.)) / norm_w, 0, 0, 0,
       (dt * (state.angular_velocity_r.x * (state.direction_w.k() * state.angular_velocity_r.x
           + state.direction_w.w() * state.angular_velocity_r.y
           - state.direction_w.i() * state.angular_velocity_r.z) * norm_w * cos(norm_w / 2.) -
           (2 * state.direction_w.w() * state.angular_velocity_r.x * state.angular_velocity_r.y
               + state.direction_w.j() * state.angular_velocity_r.x
                   * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                       + pow(state.angular_velocity_r.z, 2)) - 2
               * (state.direction_w.i() * state.angular_velocity_r.x * state.angular_velocity_r.z
                   + state.direction_w.k()
                       * (pow(state.angular_velocity_r.y, 2) + pow(state.angular_velocity_r.z, 2))))
               * sin(norm_w / 2.))) /
           (2. * pow(pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                         + pow(state.angular_velocity_r.z, 2), 1.5)), (dt
          * (state.angular_velocity_r.y * (state.direction_w.k() * state.angular_velocity_r.x
              + state.direction_w.w() * state.angular_velocity_r.y
              - state.direction_w.i() * state.angular_velocity_r.z) * norm_w *
              cos(norm_w / 2.) + (2 * state.direction_w.w()
              * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.z, 2))
              - state.angular_velocity_r.y * (2 * state.direction_w.k() * state.angular_velocity_r.x
                  - 2 * state.direction_w.i() * state.angular_velocity_r.z + state.direction_w.j()
                  * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                      + pow(state.angular_velocity_r.z, 2)))) *
              sin(norm_w / 2.))) / (2. * pow(
          pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
              + pow(state.angular_velocity_r.z, 2), 1.5)),
       (dt * (state.angular_velocity_r.z * (state.direction_w.k() * state.angular_velocity_r.x
           + state.direction_w.w() * state.angular_velocity_r.y
           - state.direction_w.i() * state.angular_velocity_r.z) * norm_w * cos(norm_w / 2.) -
           (2 * state.direction_w.i()
               * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2))
               + state.angular_velocity_r.z
                   * (2 * state.direction_w.k() * state.angular_velocity_r.x
                       + 2 * state.direction_w.w() * state.angular_velocity_r.y
                       + state.direction_w.j() * (pow(state.angular_velocity_r.x, 2)
                           + pow(state.angular_velocity_r.y, 2)
                           + pow(state.angular_velocity_r.z, 2)))) * sin(norm_w / 2.))) /
           (2. * pow(pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                         + pow(state.angular_velocity_r.z, 2), 1.5))},
      {0, 0, 0,
       (dt * state.angular_velocity_r.z * sin(norm_w / 2.)) / norm_w,
       (dt * state.angular_velocity_r.y * sin(norm_w / 2.)) / norm_w,
       -((dt * state.angular_velocity_r.x * sin(norm_w / 2.)) / norm_w), dt * cos(norm_w / 2.), 0,
       0, 0,
       -(dt * (state.angular_velocity_r.x * (state.direction_w.j() * state.angular_velocity_r.x
           - state.direction_w.i() * state.angular_velocity_r.y
           - state.direction_w.w() * state.angular_velocity_r.z) * norm_w * cos(norm_w / 2.) +
           (state.direction_w.k() * state.angular_velocity_r.x
               * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                   + pow(state.angular_velocity_r.z, 2)) + 2
               * (state.direction_w.i() * state.angular_velocity_r.x * state.angular_velocity_r.y
                   + state.direction_w.w() * state.angular_velocity_r.x * state.angular_velocity_r.z
                   + state.direction_w.j()
                       * (pow(state.angular_velocity_r.y, 2) + pow(state.angular_velocity_r.z, 2))))
               * sin(norm_w / 2.))) /
           (2. * pow(pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                         + pow(state.angular_velocity_r.z, 2), 1.5)), -(dt
          * (-(state.angular_velocity_r.y * (-(state.direction_w.j() * state.angular_velocity_r.x)
              + state.direction_w.i() * state.angular_velocity_r.y
              + state.direction_w.w() * state.angular_velocity_r.z) * norm_w *
              cos(norm_w / 2.)) + (-2 * state.direction_w.i()
              * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.z, 2)) +
              state.angular_velocity_r.y * (-2 * state.direction_w.j() * state.angular_velocity_r.x
                  + 2 * state.direction_w.w() * state.angular_velocity_r.z + state.direction_w.k()
                  * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                      + pow(state.angular_velocity_r.z, 2)))) * sin(norm_w / 2.))) /
          (2. * pow(pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                        + pow(state.angular_velocity_r.z, 2), 1.5)), -(dt
          * (-(state.angular_velocity_r.z * (-(state.direction_w.j() * state.angular_velocity_r.x)
              + state.direction_w.i() * state.angular_velocity_r.y
              + state.direction_w.w() * state.angular_velocity_r.z) * norm_w *
              cos(norm_w / 2.)) + (-2 * state.direction_w.w()
              * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)) +
              state.angular_velocity_r.z * (-2 * state.direction_w.j() * state.angular_velocity_r.x
                  + 2 * state.direction_w.i() * state.angular_velocity_r.y + state.direction_w.k()
                  * (pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                      + pow(state.angular_velocity_r.z, 2)))) * sin(norm_w / 2.))) /
          (2. * pow(pow(state.angular_velocity_r.x, 2) + pow(state.angular_velocity_r.y, 2)
                        + pow(state.angular_velocity_r.z, 2), 1.5))},
      {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
  };
  //todo: think about removing .clone()
  cout<<result_array[3][3]<<endl;
  Mat result = Mat(13, 13, CV_64F, &result_array).clone();
  return result;
}


//200 x 313
//2N x 3N+13
Mat H_t_Jacobian_of_observations(const StateMean &s, const Mat &camIntrinsics) {
  int N_points = s.feature_positions_w.size();
  Mat result(2 * N_points, 13 + 3 * N_points, CV_64F, double(0));
  for (int i_pt = 0; i_pt < s.feature_positions_w.size(); ++i_pt) {
    double q1 = s.direction_w.w();
    double q2 = s.direction_w.i();
    double q3 = s.direction_w.j();
    double q4 = s.direction_w.k();
    double r1 = s.position_w.x;
    double r2 = s.position_w.y;
    double r3 = s.position_w.z;
    double y1 = s.feature_positions_w[i_pt].x;
    double y2 = s.feature_positions_w[i_pt].y;
    double y3 = s.feature_positions_w[i_pt].z;
    double alpha_x = camIntrinsics.at<double>(0, 0);
    double alpha_y = camIntrinsics.at<double>(1, 1);
    double norm_q_squared = q1*q1 + q2*q2 + q3*q3 + q4*q4;
    double dhi_over_dx_cam_array[2][7] = {
            {-(((norm_q_squared)*
                (2*q1*q2*(-r2 + y2) + 2*q3*q4*(-r2 + y2) +
                    pow(q1,2)*(r3 - y3) +
                    pow(q3,2)*(r3 - y3) -
                    (pow(q2,2) + pow(q4,2))*(r3 - y3))*alpha_x)/
                pow(2*q3*q4*r2 - pow(q3,2)*r3 +
                    pow(q4,2)*r3 + 2*q2*q4*(r1 - y1) -
                    2*q3*q4*y2 + 2*q1*
                    (q3*(r1 - y1) + q2*(-r2 + y2)) +
                    pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                    pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2)),
             (-2*(norm_q_squared)*(q1*(q2*(r1 - y1) + q4*(r3 - y3)) +
                 q3*(q4*(r1 - y1) + q2*(-r3 + y3)))*alpha_x)/
                 pow(2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2),
             ((norm_q_squared)*(pow(q1,2)*(r1 - y1) +
                 (q3 - q4)*(q3 + q4)*(r1 - y1) +
                 pow(q2,2)*(-r1 + y1) + 2*q1*q4*(r2 - y2) +
                 2*q2*q3*(-r2 + y2))*alpha_x)/
                 pow(2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2),
             ((-2*(2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                 2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                 2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                 pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                 pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3))*
                 (q1*(r1 - y1) + q4*(r2 - y2) + q3*(-r3 + y3)) -
                 2*(q3*(r1 - y1) + q2*(-r2 + y2) + q1*(r3 - y3))*
                     ((pow(q3,2) + pow(q4,2))*(r1 - y1) +
                         pow(q1,2)*(-r1 + y1) +
                         pow(q2,2)*(-r1 + y1) +
                         2*q1*(q4*(-r2 + y2) + q3*(r3 - y3)) +
                         2*q2*(-(q3*r2) - q4*r3 + q3*y2 + q4*y3)))*alpha_x)/
                 pow(2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2),
             ((-2*(q2*(r1 - y1) + q3*(r2 - y2) + q4*(r3 - y3))*
                 (2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3)) -
                 2*(q4*(r1 - y1) + q1*(-r2 + y2) + q2*(-r3 + y3))*
                     ((pow(q3,2) + pow(q4,2))*(r1 - y1) +
                         pow(q1,2)*(-r1 + y1) +
                         pow(q2,2)*(-r1 + y1) +
                         2*q1*(q4*(-r2 + y2) + q3*(r3 - y3)) +
                         2*q2*(-(q3*r2) - q4*r3 + q3*y2 + q4*y3)))*alpha_x)/
                 pow(2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2),
             ((2*(q3*(r1 - y1) + q2*(-r2 + y2) + q1*(r3 - y3))*
                 (2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3)) -
                 2*(q1*(r1 - y1) + q4*(r2 - y2) + q3*(-r3 + y3))*
                     ((pow(q3,2) + pow(q4,2))*(r1 - y1) +
                         pow(q1,2)*(-r1 + y1) +
                         pow(q2,2)*(-r1 + y1) +
                         2*q1*(q4*(-r2 + y2) + q3*(r3 - y3)) +
                         2*q2*(-(q3*r2) - q4*r3 + q3*y2 + q4*y3)))*alpha_x)/
                 pow(2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2),
             ((2*(q4*(r1 - y1) + q1*(-r2 + y2) + q2*(-r3 + y3))*
                 (2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3)) -
                 2*(q2*(r1 - y1) + q3*(r2 - y2) + q4*(r3 - y3))*
                     ((pow(q3,2) + pow(q4,2))*(r1 - y1) +
                         pow(q1,2)*(-r1 + y1) +
                         pow(q2,2)*(-r1 + y1) +
                         2*q1*(q4*(-r2 + y2) + q3*(r3 - y3)) +
                         2*q2*(-(q3*r2) - q4*r3 + q3*y2 + q4*y3)))*alpha_x)/
                 pow(2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2)},
            {(2*(norm_q_squared)*(q2*
                (q4*(-r2 + y2) + q3*(r3 - y3)) +
                q1*(q3*(r2 - y2) + q4*(r3 - y3)))*alpha_y)/
                pow(2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                    2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                    2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                    pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                    pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2),
             -(((norm_q_squared)*
                 (2*q1*q3*(r1 - y1) + 2*q2*q4*(-r1 + y1) +
                     pow(q1,2)*(r3 - y3) +
                     pow(q2,2)*(r3 - y3) -
                     (pow(q3,2) + pow(q4,2))*(r3 - y3))*alpha_y)/
                 pow(2*q3*q4*r2 - pow(q3,2)*r3 +
                     pow(q4,2)*r3 + 2*q2*q4*(r1 - y1) -
                     2*q3*q4*y2 + 2*q1*
                     (q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2)),
             ((norm_q_squared)*(2*q2*q3*(-r1 + y1) +
                 2*q1*q4*(-r1 + y1) + pow(q1,2)*(r2 - y2) +
                 pow(q2,2)*(r2 - y2) -
                 (pow(q3,2) + pow(q4,2))*(r2 - y2))*alpha_y)/
                 pow(2*q3*q4*r2 - pow(q3,2)*r3 + pow(q4,2)*r3 +
                     2*q2*q4*(r1 - y1) - 2*q3*q4*y2 +
                     2*q1*(q3*(r1 - y1) + q2*(-r2 + y2)) +
                     pow(q1,2)*(r3 - y3) + pow(q3,2)*y3 -
                     pow(q4,2)*y3 + pow(q2,2)*(-r3 + y3),2),
             ((2*(q3*(r1 - y1) + q2*(-r2 + y2) + q1*(r3 - y3))*
                 (pow(q3,2)*r2 - pow(q4,2)*r2 + 2*q3*q4*r3 +
                     2*q1*q4*(-r1 + y1) + pow(q1,2)*(r2 - y2) -
                     pow(q3,2)*y2 + pow(q4,2)*y2 +
                     pow(q2,2)*(-r2 + y2) +
                     2*q2*(q3*(r1 - y1) + q1*(r3 - y3)) -
                     2*q3*q4*y3) +
                 2*(q4*(-r1 + y1) + q1*(r2 - y2) + q2*(r3 - y3))*
                     (-2*q3*q4*r2 + pow(q3,2)*r3 -
                         pow(q4,2)*r3 + 2*q2*q4*(-r1 + y1) +
                         2*q1*(q3*(-r1 + y1) + q2*(r2 - y2)) +
                         2*q3*q4*y2 + pow(q2,2)*(r3 - y3) -
                         pow(q3,2)*y3 + pow(q4,2)*y3 +
                         pow(q1,2)*(-r3 + y3)))*alpha_y)/
                 pow(-2*q3*q4*r2 + pow(q3,2)*r3 -
                     pow(q4,2)*r3 + 2*q2*q4*(-r1 + y1) +
                     2*q1*(q3*(-r1 + y1) + q2*(r2 - y2)) + 2*q3*q4*y2 +
                     pow(q2,2)*(r3 - y3) - pow(q3,2)*y3 +
                     pow(q4,2)*y3 + pow(q1,2)*(-r3 + y3),2),
             ((-2*(q4*(-r1 + y1) + q1*(r2 - y2) + q2*(r3 - y3))*
                 (pow(q3,2)*r2 - pow(q4,2)*r2 + 2*q3*q4*r3 +
                     2*q1*q4*(-r1 + y1) + pow(q1,2)*(r2 - y2) -
                     pow(q3,2)*y2 + pow(q4,2)*y2 +
                     pow(q2,2)*(-r2 + y2) +
                     2*q2*(q3*(r1 - y1) + q1*(r3 - y3)) -
                     2*q3*q4*y3) +
                 2*(q3*(r1 - y1) + q2*(-r2 + y2) + q1*(r3 - y3))*
                     (-2*q3*q4*r2 + pow(q3,2)*r3 -
                         pow(q4,2)*r3 + 2*q2*q4*(-r1 + y1) +
                         2*q1*(q3*(-r1 + y1) + q2*(r2 - y2)) +
                         2*q3*q4*y2 + pow(q2,2)*(r3 - y3) -
                         pow(q3,2)*y3 + pow(q4,2)*y3 +
                         pow(q1,2)*(-r3 + y3)))*alpha_y)/
                 pow(-2*q3*q4*r2 + pow(q3,2)*r3 -
                     pow(q4,2)*r3 + 2*q2*q4*(-r1 + y1) +
                     2*q1*(q3*(-r1 + y1) + q2*(r2 - y2)) + 2*q3*q4*y2 +
                     pow(q2,2)*(r3 - y3) - pow(q3,2)*y3 +
                     pow(q4,2)*y3 + pow(q1,2)*(-r3 + y3),2),
             ((-2*(q1*(-r1 + y1) + q4*(-r2 + y2) + q3*(r3 - y3))*
                 (pow(q3,2)*r2 - pow(q4,2)*r2 + 2*q3*q4*r3 +
                     2*q1*q4*(-r1 + y1) + pow(q1,2)*(r2 - y2) -
                     pow(q3,2)*y2 + pow(q4,2)*y2 +
                     pow(q2,2)*(-r2 + y2) +
                     2*q2*(q3*(r1 - y1) + q1*(r3 - y3)) -
                     2*q3*q4*y3) +
                 2*(q2*(r1 - y1) + q3*(r2 - y2) + q4*(r3 - y3))*
                     (-2*q3*q4*r2 + pow(q3,2)*r3 -
                         pow(q4,2)*r3 + 2*q2*q4*(-r1 + y1) +
                         2*q1*(q3*(-r1 + y1) + q2*(r2 - y2)) +
                         2*q3*q4*y2 + pow(q2,2)*(r3 - y3) -
                         pow(q3,2)*y3 + pow(q4,2)*y3 +
                         pow(q1,2)*(-r3 + y3)))*alpha_y)/
                 pow(-2*q3*q4*r2 + pow(q3,2)*r3 -
                     pow(q4,2)*r3 + 2*q2*q4*(-r1 + y1) +
                     2*q1*(q3*(-r1 + y1) + q2*(r2 - y2)) + 2*q3*q4*y2 +
                     pow(q2,2)*(r3 - y3) - pow(q3,2)*y3 +
                     pow(q4,2)*y3 + pow(q1,2)*(-r3 + y3),2),
             ((2*(q1*(-r1 + y1) + q4*(-r2 + y2) + q3*(r3 - y3))*
                 (-2*q3*q4*r2 + pow(q3,2)*r3 -
                     pow(q4,2)*r3 + 2*q2*q4*(-r1 + y1) +
                     2*q1*(q3*(-r1 + y1) + q2*(r2 - y2)) +
                     2*q3*q4*y2 + pow(q2,2)*(r3 - y3) -
                     pow(q3,2)*y3 + pow(q4,2)*y3 +
                     pow(q1,2)*(-r3 + y3)) -
                 2*(pow(q3,2)*r2 - pow(q4,2)*r2 +
                     2*q3*q4*r3 + 2*q1*q4*(-r1 + y1) +
                     pow(q1,2)*(r2 - y2) - pow(q3,2)*y2 +
                     pow(q4,2)*y2 + pow(q2,2)*(-r2 + y2) +
                     2*q2*(q3*(r1 - y1) + q1*(r3 - y3)) -
                     2*q3*q4*y3)*
                     (q2*(-r1 + y1) + q3*(-r2 + y2) + q4*(-r3 + y3)))
                 *alpha_y)/
                 pow(-2*q3*q4*r2 + pow(q3,2)*r3 -
                     pow(q4,2)*r3 + 2*q2*q4*(-r1 + y1) +
                     2*q1*(q3*(-r1 + y1) + q2*(r2 - y2)) + 2*q3*q4*y2 +
                     pow(q2,2)*(r3 - y3) - pow(q3,2)*y3 +
                     pow(q4,2)*y3 + pow(q1,2)*(-r3 + y3),2)}};
    double dhi_over_dz_array[2][3] =
        {
            {alpha_x * (pow(q1, 2) + pow(q2, 2) + pow(q3, 2) +
                pow(q4, 2)) * (2 * q1 * q2 * (-r2 + y2) +
                2 * q3 * q4 * (-r2 + y2) + (r3 - y3) * pow(q1, 2) +
                (r3 - y3) * pow(q3, 2) -
                (r3 - y3) * (pow(q2, 2) + pow(q4, 2))) *
                pow(2 * q3 * q4 * r2 + 2 * q2 * q4 * (r1 - y1) - 2 * q3 * q4 * y2 +
                    2 * q1 * (q3 * (r1 - y1) + q2 * (-r2 + y2)) +
                    (r3 - y3) * pow(q1, 2) + (-r3 + y3) * pow(q2, 2) -
                    r3 * pow(q3, 2) + y3 * pow(q3, 2) + r3 * pow(q4, 2) -
                    y3 * pow(q4, 2), -2),
             2 * alpha_x * (q1 * (q2 * (r1 - y1) + q4 * (r3 - y3)) +
                 q3 * (q4 * (r1 - y1) + q2 * (-r3 + y3))) *
                 (pow(q1, 2) + pow(q2, 2) + pow(q3, 2) + pow(q4, 2)) *
                 pow(2 * q3 * q4 * r2 + 2 * q2 * q4 * (r1 - y1) - 2 * q3 * q4 * y2 +
                     2 * q1 * (q3 * (r1 - y1) + q2 * (-r2 + y2)) +
                     (r3 - y3) * pow(q1, 2) + (-r3 + y3) * pow(q2, 2) -
                     r3 * pow(q3, 2) + y3 * pow(q3, 2) + r3 * pow(q4, 2) -
                     y3 * pow(q4, 2), -2),
             -(alpha_x * ((q3 - q4) * (q3 + q4) * (r1 - y1) +
                 2 * q1 * q4 * (r2 - y2) + 2 * q2 * q3 * (-r2 + y2) +
                 (r1 - y1) * pow(q1, 2) + (-r1 + y1) * pow(q2, 2)) *
                 (pow(q1, 2) + pow(q2, 2) + pow(q3, 2) + pow(q4, 2)) *
                 pow(2 * q3 * q4 * r2 + 2 * q2 * q4 * (r1 - y1) - 2 * q3 * q4 * y2 +
                     2 * q1 * (q3 * (r1 - y1) + q2 * (-r2 + y2)) +
                     (r3 - y3) * pow(q1, 2) + (-r3 + y3) * pow(q2, 2) -
                     r3 * pow(q3, 2) + y3 * pow(q3, 2) + r3 * pow(q4, 2) -
                     y3 * pow(q4, 2), -2))},
            {-2 * alpha_y * (q2 * (q4 * (-r2 + y2) + q3 * (r3 - y3)) +
                q1 * (q3 * (r2 - y2) + q4 * (r3 - y3))) *
                (pow(q1, 2) + pow(q2, 2) + pow(q3, 2) + pow(q4, 2)) *
                pow(2 * q3 * q4 * r2 + 2 * q2 * q4 * (r1 - y1) - 2 * q3 * q4 * y2 +
                    2 * q1 * (q3 * (r1 - y1) + q2 * (-r2 + y2)) +
                    (r3 - y3) * pow(q1, 2) + (-r3 + y3) * pow(q2, 2) -
                    r3 * pow(q3, 2) + y3 * pow(q3, 2) + r3 * pow(q4, 2) -
                    y3 * pow(q4, 2), -2),
             alpha_y * (pow(q1, 2) + pow(q2, 2) + pow(q3, 2) + pow(q4, 2)) *
                 (2 * q1 * q3 * (r1 - y1) + 2 * q2 * q4 * (-r1 + y1) +
                     (r3 - y3) * pow(q1, 2) + (r3 - y3) * pow(q2, 2) -
                     (r3 - y3) * (pow(q3, 2) + pow(q4, 2))) *
                 pow(2 * q3 * q4 * r2 + 2 * q2 * q4 * (r1 - y1) - 2 * q3 * q4 * y2 +
                     2 * q1 * (q3 * (r1 - y1) + q2 * (-r2 + y2)) +
                     (r3 - y3) * pow(q1, 2) + (-r3 + y3) * pow(q2, 2) -
                     r3 * pow(q3, 2) + y3 * pow(q3, 2) + r3 * pow(q4, 2) -
                     y3 * pow(q4, 2), -2),
             -(alpha_y * (pow(q1, 2) + pow(q2, 2) + pow(q3, 2) +
                 pow(q4, 2)) * (2 * q2 * q3 * (-r1 + y1) +
                 2 * q1 * q4 * (-r1 + y1) + (r2 - y2) * pow(q1, 2) +
                 (r2 - y2) * pow(q2, 2) -
                 (r2 - y2) * (pow(q3, 2) + pow(q4, 2))) *
                 pow(2 * q3 * q4 * r2 + 2 * q2 * q4 * (r1 - y1) - 2 * q3 * q4 * y2 +
                     2 * q1 * (q3 * (r1 - y1) + q2 * (-r2 + y2)) +
                     (r3 - y3) * pow(q1, 2) + (-r3 + y3) * pow(q2, 2) -
                     r3 * pow(q3, 2) + y3 * pow(q3, 2) + r3 * pow(q4, 2) -
                     y3 * pow(q4, 2), -2))
            }
        };
    Mat dhi_over_dx_cam = Mat(2, 7, CV_64F, &dhi_over_dx_cam_array);
    Mat dhi_over_dz = Mat(2, 3, CV_64F, &dhi_over_dz_array);
    Mat dcam_area = result(Rect(0, 2*i_pt, 7, 2));
    dhi_over_dx_cam.copyTo(dcam_area);
    Mat dz_area = result(Rect(13 + 3 * i_pt, 2*i_pt, 3, 2));
    dhi_over_dz.copyTo(dz_area);
  }
  return result;
}

#endif //MY_SLAM_JACOBIANS_H
