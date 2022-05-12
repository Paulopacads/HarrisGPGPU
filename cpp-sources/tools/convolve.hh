#ifndef CONVOLVE_HH
#define CONVOLVE_HH

#include "../construct/matrix.hh"

template <typename num1, typename num2>
matrix<num1> *convolve(matrix<num1> *m1, matrix<num2> *m2) {
    matrix<num1> *output = new matrix<num1>(m1->rows, m1->cols);
    int x_pad = m2->rows / 2;
    int y_pad = m2->cols / 2;

    for (int i = x_pad; i < m1->rows - x_pad; i++) {
        for (int j = y_pad; j < m1->cols - y_pad; j++) {
            float sum = 0;

            for (int k = 0; k < m2->rows; k++) {
                for (int l = 0; l < m2->cols; l++) {
                    sum += (*m2)[k + l * m2->rows]
                    * (*m1)[i - x_pad + k + (j - y_pad + l) * m1->rows];
                }
            }
            (*output)[i + j * m1->rows] = sum;
        }
    }
    return output;
}

#endif /* CONVOLVE_HH */