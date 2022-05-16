#pragma once

#include "../construct/matrix.hh"

#include <cstdio>

matrix<float> *convolve(uint8_t *m1, matrix<float> *m2, int m1_rows, int m1_cols);
matrix<float> *convolve(matrix<float> *m1, float *m2, int m2_rows, int m2_cols);