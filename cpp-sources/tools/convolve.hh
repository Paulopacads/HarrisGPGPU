#ifndef CONVOLVE_HH
#define CONVOLVE_HH

#include "../construct/matrix.hh"

template <typename num1, typename num2>
matrix<float> *convolve(matrix<num1> *m1, matrix<num2> *m2) {
    matrix<float> *output = new matrix<float>(m1->rows, m1->cols);
    int x_pad = m2->rows / 2;
    int y_pad = m2->cols / 2;

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            int sum = 0.0;
            float conv = 0;

            for (int k = 0; k < m2->rows; k++) {
                int x_pos = i - x_pad + k;
                for (int l = 0; l < m2->cols; l++) {
                    int y_pos = j - y_pad + l;
                    num1 value = 128;
                    if (x_pos >= 0 && y_pos >= 0 && x_pos < m1->rows && y_pos < m1->cols)
                        value = (*m1)[x_pos + y_pos * m1->rows];
                    sum += (*m2)[k + l * m2->rows];
                    conv += (*m2)[k + l * m2->rows] * value;
                }
            }
            (*output)[i + j * m1->rows] = sum;
        }
    }
    return output;
}

#endif /* CONVOLVE_HH */