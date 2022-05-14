#include <cstdint>
#include <cstdlib>
#include <sys/types.h>

#include <cstdio>
#include <cassert>

#include "matrix.hh"

template <typename number>
matrix<number>::matrix(int rows, int cols) : rows(rows), cols(cols) {
  values = (number *)malloc(sizeof(number) * rows * cols);
}

template <typename number>
matrix<number>::matrix(int rows, int cols, number *values)
    : rows(rows), cols(cols), values(values) {}

template <typename number> matrix<number>::~matrix() { free(values); }

template <typename number> matrix<number> *matrix<number>::operator*(float n) {
  matrix *output = new matrix(rows, cols);
  for (int i = 0; i < rows * cols; i++) {
    output->values[i] = values[i] * n;
  }
  return output;
}


template <typename number> matrix<number> *matrix<number>::operator/(float n) {
  matrix *output = new matrix(rows, cols);
  for (int i = 0; i < rows * cols; i++) {
    output->values[i] = values[i] / n;
  }
  return output;
}

template <typename number>
matrix<number> *matrix<number>::operator>(float n) {
    matrix *output = new matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        if ((*output)[i] > n)
            (*output)[i] = true;
        else
            (*output)[i] = false;
    }
    return output;
}

template <typename number>
number matrix<number>::min() {
    number output = values[0];
    for (int i = 1; i < rows * cols; i++) {
        if (values[i] < output)
            output = values[i];
    }
    return output;
}

template <typename number>
number matrix<number>::max() {
    number output = values[0];
    for (int i = 1; i < rows * cols; i++) {
        if (values[i] > output)
            output = values[i];
    }
    return output;
}


template <typename number>
matrix<number> *matrix<number>::transpose() {
    matrix *output = new matrix(cols, rows);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output->values[j + i * output->rows] = values[i + j * rows];
        }
    }
    return output;
}

template <typename number> matrix<number> *matrix<number>::operator+(float n) {
  matrix *output = new matrix(rows, cols);

  for (int i = 0; i < rows * cols; i++) {
    output->values[i] = values[i] + n;
  }
  return output;
}

template <typename number> number &matrix<number>::operator[](int i) {
  return values[i];
}

template <typename num1, typename num2>
matrix<num1> *mat_multiply_element_wise(matrix<num1> *mat1,
                                        matrix<num2> *mat2) {
  assert(mat1->rows == mat2->rows && mat1->cols == mat2->cols);

  matrix<num1> *newMat = new matrix<num1>(mat1->rows, mat1->cols);

  int size = mat1->rows * mat1->cols;

  for (int i = 0; i < size; i++) {
    (*newMat)[i] = (*mat1)[i] * (*mat2)[i];
  }

  return newMat;
}

template <typename num1, typename num2>
matrix<num1> *mat_diff_element_wise(matrix<num1> *mat1, matrix<num2> *mat2) {
  //assert(mat1->rows == mat2->rows && mat1->cols == mat2->cols);

  matrix<num1> *newMat = new matrix<num1>(mat1->rows, mat1->cols);

  int size = mat1->rows * mat1->cols;

  for (int i = 0; i < size; i++) {
    (*newMat)[i] = (*mat1)[i] - (*mat2)[i];
  }

  return newMat;
}

template <typename num1, typename num2>
matrix<num1> *mat_divide_element_wise(matrix<num1> *mat1, matrix<num2> *mat2) {
  //assert(mat1->rows == mat2->rows && mat1->cols == mat2->cols);

  matrix<num1> *newMat = new matrix<num1>(mat1->rows, mat1->cols);

  int size = mat1->rows * mat1->cols;

  for (int i = 0; i < size; i++) {
    (*newMat)[i] = (*mat1)[i] / (*mat2)[i];
  }

  return newMat;
}

template <typename num1, typename num2>
matrix<num1> *mat_add_element_wise(matrix<num1> *mat1, matrix<num2> *mat2) {
  //assert(mat1->rows == mat2->rows && mat1->cols == mat2->cols);

  matrix<num1> *newMat = new matrix<num1>(mat1->rows, mat1->cols);

  int size = mat1->rows * mat1->cols;

  for (int i = 0; i < size; i++) {
    (*newMat)[i] = (*mat1)[i] + (*mat2)[i];
  }

  return newMat;
}

template matrix<float>* mat_add_element_wise<float, float>(matrix<float> *mat1, matrix<float> *mat2);
template matrix<float>* mat_diff_element_wise<float, float>(matrix<float> *mat1, matrix<float> *mat2);
template matrix<float>* mat_multiply_element_wise<float, float>(matrix<float> *mat1, matrix<float> *mat2);
template matrix<float>* mat_divide_element_wise<float, float>(matrix<float> *mat1, matrix<float> *mat2);

template class matrix<float>;
template class matrix<uint8_t>;
template class matrix<bool>;
