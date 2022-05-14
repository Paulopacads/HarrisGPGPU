#include "construct/matrix.hh"
#include "derivatives.hh"
#include "tools/convolve.hh"
#include <cmath>
#include <cstdint>
#include <stdlib.h>

#include <cstdio>

using namespace std;

matrix<float> *mgridY(int size) {
  matrix<float> *meshGrid = new matrix<float>(size * 2 + 1, size * 2 + 1);

  for (int i = -size; i <= size; i++) {
    for (int j = -size; j <= size; j++) {
      (*meshGrid)[(size + i) * (size * 2 + 1) + (size + j)] = i;
    }
  }
  return meshGrid;
}

matrix<float> *mgridX(int size) {
  matrix<float> *meshGrid = new matrix<float>(size * 2 + 1, size * 2 + 1);

  for (int i = -size; i <= size; i++) {
    for (int j = -size; j <= size; j++) {
      (*meshGrid)[(size + i) * (size * 2 + 1) + (size + j)] = j;
    }
  }
  return meshGrid;
}

matrix<float> *gauss_kernel(int size) {
  matrix<float> *yGrid = mgridY(size);
  matrix<float> *xGrid = mgridX(size);

  float sigma = (float)size / 3;

  matrix<float> *meshGrid = new matrix<float>(size * 2 + 1, size * 2 + 1);

  for (int i = 0; i < (size * 2) + 1; i++) {
    for (int j = 0; j < (size * 2) + 1; j++) {
      int index = i * (size * 2 + 1) + j;
      float x = (*xGrid)[index];
      float y = (*yGrid)[index];
      (*meshGrid)[index] =
          exp(-(pow(x, 2) / (2 * pow(sigma, 2)) + pow(y, 2) / (2 * pow(sigma, 2))));
    }
  }

  delete yGrid;
  delete xGrid;

  return meshGrid;
}

tuple_matrix<float> gauss_derivative_kernels(int size) {
  matrix<float> *xGrid = mgridX(size);
  matrix<float> *yGrid = mgridY(size);

  matrix<float> *gx = new matrix<float>(size * 2 + 1, size * 2 + 1);
  matrix<float> *gy = new matrix<float>(size * 2 + 1, size * 2 + 1);

  float sigma = (float) size / 3;

  for (int i = 0; i < size * 2 + 1; i++) {
    for (int j = 0; j < size * 2 + 1; j++) {
      int index = i * (size * 2 + 1) + j;
      float x = (*xGrid)[index];
      float y = (*yGrid)[index];
      float val = exp(-(pow(x, 2) / (2 * pow(sigma, 2)) + pow(y, 2) / (2 * pow(sigma, 2))));
      (*gx)[index] = -x * val;
      (*gy)[index] = -y * val;
    }
  }

  //delete yGrid;
  //delete xGrid;

  tuple_matrix<float> res{
      gx,
      gy,
  };

  return res;
}

tuple_matrix<float> gauss_derivatives(matrix<uint8_t> *img, int size) {
  tuple_matrix<float> gxy = gauss_derivative_kernels(size);

  matrix<float> *imx = convolve(img, gxy.mat1);
  matrix<float> *imy = convolve(img, gxy.mat2);

  tuple_matrix<float> res{
      imx,
      imy,
  };

  return res;
}

matrix<float> *compute_harris_response(matrix<uint8_t> *img) {
  int derivativeKernelSize = 1;
  int opening_size = 1;

  tuple_matrix<float> tupleImxy =
      gauss_derivatives(img, derivativeKernelSize);

  matrix<float> *gauss = gauss_kernel(opening_size);

  matrix<float> *imxx =
      mat_multiply_element_wise(tupleImxy.mat1, tupleImxy.mat1);
  matrix<float> *imyy =
      mat_multiply_element_wise(tupleImxy.mat2, tupleImxy.mat2);
  matrix<float> *imxy =
      mat_multiply_element_wise(tupleImxy.mat1, tupleImxy.mat2);

  matrix<float> *wxx = convolve(imxx, gauss);
  matrix<float> *wxy = convolve(imxy, gauss);
  matrix<float> *wyy = convolve(imyy, gauss);

  matrix<float> *wdet = mat_diff_element_wise(
      mat_multiply_element_wise(wxx, wyy), mat_multiply_element_wise(wxy, wxy));

  matrix<float> *wtr = mat_add_element_wise(wxx, wyy);

  matrix<float> *wtr1 =  *wtr + 1;
  
  matrix<float> *res = mat_divide_element_wise(wdet, wtr1);

  /*
  delete gauss;
  delete imxx;
  delete imyy;
  delete imxy;
  delete wxx;
  delete wyy;
  delete wxy;
  delete wdet;
  delete wtr;

  return res;
  */
  return res;
}