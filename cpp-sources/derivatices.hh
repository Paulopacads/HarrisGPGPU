#pragma once

struct tuple_array {
    float* arr1;
    float* arr2;
};

float* gauss_kernel(int size); 
tuple_array gauss_derivative_kernels(int size);
tuple_array gauss_derivatives(int* img, int size);