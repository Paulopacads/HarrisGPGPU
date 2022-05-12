#pragma once

#include "construct/matrix.hh"
#include <cstdint>

template <typename T>
struct tuple_matrix {
    matrix<T>* mat1;
    matrix<T>* mat2;
};

matrix<float>* gauss_kernel(int size); 
tuple_matrix<float> gauss_derivative_kernels(int size);
tuple_matrix<uint8_t> gauss_derivatives(int* img, int size);