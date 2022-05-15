#ifndef MORPH_HH
#define MORPH_HH

#include <cmath>
#include <stdexcept>
#include "uchar.h"

#include "../construct/matrix.hh"

template <typename num1, typename num2>
matrix<num1> *dilate(matrix<num1> *m1, matrix<num2> *m2) {
    matrix<num1> *output = new matrix<num1>(m1->rows, m1->cols);
    int x_pad = m2->rows / 2;
    int y_pad = m2->cols / 2;

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            num1 max = 0;

            for (int k = 0; k < m2->rows; k++) {
                for (int l = 0; l < m2->cols; l++) {
                    if ((*m2)[k * m2->cols + l] && i - x_pad + k >= 0
                    && i - x_pad + k < m1->rows && j - y_pad + l >= 0
                    && j - y_pad + l < m1->cols) {
                        num1 tmp = (*m1)[(i - x_pad + k) * m1->cols + j - y_pad + l];
                        if (tmp > max)
                            max = tmp;
                    }
                }
            }
            (*output)[i * m1->cols + j] = max;
        }
    }
    return output;
}

template <typename num1, typename num2>
matrix<num1> *erode(matrix<num1> *m1, matrix<num2> *m2) {
    matrix<num1> *output = new matrix<num1>(m1->rows, m1->cols);
    int y_pad = m2->rows / 2;
    int x_pad = m2->cols / 2;

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            num1 min = 1;

            for (int k = 0; k < m2->rows; k++) {
                int y_pos = i - y_pad + k;
                for (int l = 0; l < m2->cols; l++) {
                    int x_pos = j - x_pad + l;
                    if ((*m2)[k * m2->cols + l] && y_pos >= 0 && y_pos < m1->rows
                    && x_pos >= 0 && x_pos < m1->cols) {
                        num1 tmp = (*m1)[y_pos * m1->cols + x_pos];
                        if (tmp < min)
                            min = tmp;
                    }
                }
            }
            (*output)[i * m1->cols + j] = min;
        }
    }
    return output;
}

matrix<bool> *getStructuringElement(int rows, int cols);

matrix<bool> *bubble2maskeroded(matrix<uint8_t> *img_gray, int border=10);

#endif /* MORPH_HH */