#pragma once

#include <cstdint>

#include "derivatives.hh"

matrix<float> *compute_harris_response(matrix<uint8_t> *img);

matrix<int> *detect_harris_points(matrix<uint8_t> *image_gray, int max_keypoints=30, 
int min_distance=25, float threshold=0.1);