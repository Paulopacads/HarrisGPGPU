#pragma once

#include <cstdint>

template <typename number> class matrix {
private:
  void _view(int i);

public:
  int rows;
  int cols;

  number *values;

  matrix(int rows, int cols);
  matrix(int rows, int cols, number *values);

  ~matrix();

  number min();
  number max();

  matrix<number> *transpose();
  matrix<int> *non_zero_transposed();

  matrix<number> *operator*(float n);
  matrix<number> *operator/(float n);
  matrix<number> *operator+(float n);
  matrix<bool> *operator>(float n);

  number &operator[](int i);

  void view();
};

matrix<float> *mat_multiply_element_wise(matrix<float> *mat1,
                                        matrix<float> *mat2);
template <typename num1, typename num2>
matrix<num1> *mat_diff_element_wise(matrix<num1> *mat1, matrix<num2> *mat2);
template <typename num1, typename num2>
matrix<num1> *mat_divide_element_wise(matrix<num1> *mat1, matrix<num2> *mat2);
template <typename num1, typename num2>
matrix<num1> *mat_add_element_wise(matrix<num1> *mat1, matrix<num2> *mat2);

template <typename number>
void quickSort(matrix<int> *indices, matrix<number> *values, int start, int end);

void bubbleSort(matrix<int> *indices, matrix<float> *values, int n);