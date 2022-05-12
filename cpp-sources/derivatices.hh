#pragma once

#include "construct/matrix.hh"
#include <cstdint>

template <typename T> struct tuple_matrix {
  matrix<T> *mat1;
  matrix<T> *mat2;
};

matrix<float> *gauss_kernel(int size);
tuple_matrix<float> gauss_derivative_kernels(int size);
tuple_matrix<uint8_t> gauss_derivatives(matrix<uint8_t> *img, int size);
matrix<uint8_t> *compute_harris_response(matrix<uint8_t> *img);