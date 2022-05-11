#include <cstdlib>

#include "matrix.hh"

matrix::matrix(int rows, int cols)
: rows(rows), cols(cols) {
    values = (float *) malloc(sizeof(float) * rows * cols);
}

matrix::~matrix() {
    free(values);
}

matrix matrix::operator*(float n) {
    matrix output = matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        output[i] = values[i] * n;
    }
    return output;
}

float &matrix::operator[](int i) {
    return values[i];
}