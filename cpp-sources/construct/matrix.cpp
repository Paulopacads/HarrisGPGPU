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
matrix<number> *matrix<number>::operator*(float n) {
    matrix *output = new matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        output->values[i] = values[i] * n;
    }
    return output;
}

template <typename number>
matrix<number> *matrix<number>::operator/(float n) {
    matrix *output = new matrix(rows, cols);
    for (int i = 0; i < rows * cols; i++) {
        output->values[i] = values[i] / n;
    }
    return output;
}

template <typename number>
number &matrix<number>::operator[](int i) {
    return values[i];
}

template class matrix<float>;
template class matrix<uint8_t>;