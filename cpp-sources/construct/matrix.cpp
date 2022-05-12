#include <cstdlib>
#include <cstdint>

#include "matrix.hh"

template <typename number>
matrix<number>::matrix(int rows, int cols)
: rows(rows), cols(cols) {
    values = (number *) malloc(sizeof(number) * rows * cols);
}

template <typename number>
matrix<number>::matrix(int rows, int cols, number *values)
: rows(rows), cols(cols), values(values) {}

template <typename number>
matrix<number>::~matrix() {
    free(values);
}

template <typename number>
matrix<number> *matrix<number>::operator*(float n) {
    matrix *output = new matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        (*output)[i] = values[i] * n;
    }
    return output;
}

template <typename number>
number &matrix<number>::operator[](int i) {
    return values[i];
}

template class matrix<float>;
template class matrix<uint8_t>;