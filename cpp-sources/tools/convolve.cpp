#include "convolve.hh"

matrix *convolve(matrix *m1, matrix *m2) {
    matrix *output = new matrix(m1->rows, m1->cols);
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
            (*output)[i + j * m1->rows] = sum / (m2->rows * m2->cols);
        }
    }
    return output;
}