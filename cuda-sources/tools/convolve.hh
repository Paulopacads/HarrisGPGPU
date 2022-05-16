#pragma once

#include "../construct/matrix.hh"

#include <cstdio>

matrix<float> *convolve(uint8_t *m1, matrix<float> *m2, int m1_rows, int m1_cols);
matrix<float> *convolve(matrix<float> *m1, float *m2, int m2_rows, int m2_cols);

/*
template <typename num1, typename num2>
matrix<float> *convolve(matrix<num1> *m1, matrix<num2> *m2) {
    matrix<float> *output = new matrix<float>(m1->rows, m1->cols);
    int x_pad = m2->rows / 2;
    int y_pad = m2->cols / 2;

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            float conv = 0;

            for (int k = 0; k < m2->rows; k++) {
                int x_pos = i - x_pad + k;
                for (int l = 0; l < m2->cols; l++) {
                    int y_pos = j - y_pad + l;
                    num1 value = 0;
                    if (x_pos >= 0 && y_pos >= 0 && x_pos < m1->rows && y_pos < m1->cols)
                        value = (*m1)[x_pos * m1->cols + y_pos];
                    conv += (*m2)[k * m2->cols + l] * value;
                    //if (i == 0 && j == 1)
                    //    printf("value: %d, conv: %f\n", value, (*m2)[k * m2->cols + l]);
                }
            }
            //printf("sum: %f ", conv);
            (*output)[i * m1->cols + j] = conv;
        }
    }
    return output;
}*/