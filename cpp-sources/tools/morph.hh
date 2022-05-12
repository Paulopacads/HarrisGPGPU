#ifndef MORPH_HH
#define MORPH_HH

#include "../construct/matrix.hh"

template <typename num1, typename num2>
matrix<num1> *dilate(matrix<num1> *m1, matrix<num2> *m2) {
    matrix<num1> *output = new matrix<num1>(m1->rows, m1->cols);
    int x_pad = m2->rows / 2;
    int y_pad = m2->cols / 2;

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            int max = 0;

            for (int k = 0; k < m2->rows; k++) {
                for (int l = 0; l < m2->cols; l++) {
                    if ((*m2)[k + l * m2->rows] && i - x_pad + k >= 0
                    && i - x_pad + k < m1->rows && j - y_pad + l >= 0
                    && j - y_pad + l < m1->cols) {
                        num1 tmp =  (*m1)[i - x_pad + k + (j - y_pad + l) * m1->rows];
                        if (tmp > max)
                            max = tmp;
                    }
                }
            }
            (*output)[i + j * m1->rows] = max;
        }
    }
    return output;
}

template <typename num1, typename num2>
matrix<num1> *erode(matrix<num1> *m1, matrix<num2> *m2) {
    matrix<num1> *output = new matrix<num1>(m1->rows, m1->cols);
    int x_pad = m2->rows / 2;
    int y_pad = m2->cols / 2;

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            int min = 255;

            for (int k = 0; k < m2->rows; k++) {
                for (int l = 0; l < m2->cols; l++) {
                    if ((*m2)[k + l * m2->rows] && i - x_pad + k >= 0
                    && i - x_pad + k < m1->rows && j - y_pad + l >= 0
                    && j - y_pad + l < m1->cols) {
                        num1 tmp =  (*m1)[i - x_pad + k + (j - y_pad + l) * m1->rows];
                        if (tmp < min)
                            min = tmp;
                    }
                }
            }
            (*output)[i + j * m1->rows] = min;
        }
    }
    return output;
}

#endif /* MORPH_HH */