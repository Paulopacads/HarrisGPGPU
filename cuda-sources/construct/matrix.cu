#include <cstdint>
#include <cstdlib>
#include <sys/types.h>

#include <cstdio>
#include <cassert>
#include <iostream>

#include "matrix.hh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
matrix<bool> *matrix<number>::operator>(float n) {
    matrix<bool> *output = new matrix<bool>(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        if (values[i] > n)
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

template <typename number>
matrix<int> *matrix<number>::non_zero_transposed() {
    int new_rows = 0;
    for (int i = 0; i < rows * cols; ++i) {
        if (values[i] == true)
            new_rows++;
    }

    matrix<int> *output = new matrix<int>(new_rows, 2);
    int k = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (values[i * cols + j] == true) {
                (*output)[k++] = i;
                (*output)[k++] = j;
            }
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

__global__ void multiply_cu(float* output, float* m1, float* m2, int mat_cols) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    output[i * mat_cols + j] = m1[i * mat_cols + j] * m2[i * mat_cols + j];
}


matrix<float> *mat_multiply_element_wise(matrix<float> *mat1,
                                        matrix<float> *mat2) {
  assert(mat1->rows == mat2->rows && mat1->cols == mat2->cols);
  
  float *output;

  cudaMallocManaged(&output, mat1->rows * mat1->cols * sizeof(float));
  gpuErrchk(cudaGetLastError());

  matrix<float> *newMat = new matrix<float>(mat1->rows, mat1->cols, output);

  int tx = 24;
  int ty = 16;

  dim3 blocks(mat1->cols / tx, mat1->rows / ty);
  dim3 threads(tx, ty);
  
  multiply_cu<<<blocks, threads>>>(output, mat1->values, mat2->values, mat1->cols);
  gpuErrchk(cudaGetLastError());

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
template matrix<float>* mat_divide_element_wise<float, float>(matrix<float> *mat1, matrix<float> *mat2);

template <typename number>
int partition(matrix<int> *indices, matrix<number> *values, int start, int end)
{

    number pivot = (*values)[(*indices)[start]];

    int count = 0;
    for (int i = start + 1; i <= end; i++) {
        if ((*values)[(*indices)[i]] <= pivot)
            count++;
    }

    // Giving pivot element its correct position
    int pivotIndex = start + count;
    int swap = (*indices)[pivotIndex];
    (*indices)[pivotIndex] = (*indices)[start];
    (*indices)[start] = swap;
    //swap(arr[pivotIndex], arr[start]);

    // Sorting left and right parts of the pivot element
    int i = start, j = end;

    while (i < pivotIndex && j > pivotIndex) {

        while ((*values)[(*indices)[i]] <= pivot) {
            i++;
        }

        while ((*values)[(*indices)[j]] > pivot) {
            j--;
        }

        if (i < pivotIndex && j > pivotIndex) {
            int swap = (*indices)[i];
            (*indices)[i] = (*indices)[j];
            (*indices)[j] = swap;
            i++;
            j--;
            //swap(arr[i++], arr[j--]);
        }
    }
    return pivotIndex;
}

template <typename number>
void quickSort(matrix<int> *indices, matrix<number> *values, int start, int end)
{

    // base case
    if (start >= end)
        return;

    // partitioning the array
    int p = partition(indices, values, start, end);

    // Sorting the left part
    quickSort(indices, values, start, p - 1);

    // Sorting the right part
    quickSort(indices, values, p + 1, end);
}

template void quickSort(matrix<int> *indices, matrix<float> *values, int start, int end);

void bubbleSort(matrix<int> *indices, matrix<float> *values, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        // Last i elements are already
        // in place
        for (int j = 0; j < n - i - 1; j++)
        {
            if ((*values)[(*indices)[j]] > (*values)[(*indices)[j + 1]])
            {
                int swap = (*indices)[j];
                (*indices)[j] = (*indices)[j + 1];
                (*indices)[j + 1] = swap;
            }
        }
    }
}

template <typename number>
void matrix<number>::_view(int i) {
  if (i != 0)
    std::cout << ' ';
  std::cout << '[';
  for (int j = 0; j < 3; j++)
    std::cout << " " << values[i * cols + j];
  std::cout << " ...";
  for (int j = cols - 3; j < cols; j++)
    std::cout << " " << values[i * cols + j];
  std::cout << ']';
  if (i != rows - 1)
    std::cout << std::endl;
}

template <typename number>
void matrix<number>::view() {
  std::cout << '[';
  for (int i = 0; i < 3; i++) {
    _view(i);
  } 
  std::cout << " ..." << std::endl;
  for (int i = rows - 3; i < rows; i++) {
    _view(i);
  } 
  std::cout << ']' << std::endl;
}


template class matrix<float>;
template class matrix<uint8_t>;
template class matrix<bool>;
template class matrix<int>;