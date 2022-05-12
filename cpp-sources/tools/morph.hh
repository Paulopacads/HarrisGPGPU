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

template <typename number>
matrix<number> *getStructuringElement(int rows, int cols)
{
    int i, j;
    int r = 0, c = 0;
    double inv_r2 = 0;

    r = rows/2;
    c = cols/2;
    inv_r2 = r ? 1./((double)r*r) : 0;

    matrix<number> *elem = new matrix<number>(rows, cols);

    for( i = 0; i < rows; i++ )
    {
        // elem.data : pointer vers premier élément
        // elem.step : nombre de bytes entre le 1er élément d'un row et celui du row suivant
        // unsigned char* ptr = elem.data + i*elem.step; <--- ligne d'origine sur de opencv
        unsigned char* ptr = elem + i*sizeof(number)*cols; /// FIX ME ?
        int j1 = 0, j2 = 0;
        int dy = i - r;
        if( std::abs(dy) <= r )
        {
            int dx = c*std::sqrt((r*r - dy*dy)*inv_r2);
            j1 = std::max( c - dx, 0 );
            j2 = std::min( c + dx + 1, cols );
        }

        for( j = 0; j < j1; j++ )
            ptr[j] = 0;
        for( ; j < j2; j++ )
            ptr[j] = 1;
        for( ; j < rows; j++ )
            ptr[j] = 0;
    }

    return elem;
}

template <typename number>
matrix<number> *bubble2maskeroded(matrix<number> *img_gray, int border=10)
{
    matrix<bool> *mask = img_gray > 0;
    matrix<number> *kernel = getStructuringElement<number>(border*2, border*2);

    matrix<number> *mask_er = erode(mask, kernel);
    return mask_er > 0;
}

#endif /* MORPH_HH */