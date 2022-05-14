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

template <typename number>
matrix<number> *detect_harris_points(matrix<number> *image_gray, int max_keypoints=30,
                                     int min_distance=25, float threshold=0.1)
{
    // 1. Compute Harris corner response
    matrix<uint8_t> *harris_resp = compute_harris_response(image_gray);

    // 2. Filtering
    // 2.0 Mask init: all our filtering is performed using a mask
    matrix<bool> *detect_mask = new matrix<bool>(harris_resp->rows, harris_resp->cols);
    for (int i = 0; i < detect_mask->rows * detect_mask->cols; i++) {
        (*detect_mask)[i] = true;
    }

    // 2.1 Background and border removal
    matrix<bool> *mask_bubble = bubble2maskeroded(image_gray, min_distance);
    for (int i = 0; i < detect_mask->rows * detect_mask->cols; i++) {
        if (!((*detect_mask)[i] && (*mask_bubble)[i]))
            (*detect_mask)[i] = false;
    }

    // 2.2 Response threshold
    uint8_t min_harris_resp = harris_resp->min();
    auto new_tresh = min_harris_resp + threshold * (harris_resp->max() - min_harris_resp);

    // remove low response elements
    matrix<bool> *mask_harris = new matrix<bool>(harris_resp->rows, harris_resp->cols);
    for (int i = 0; i < mask_harris->rows * mask_harris->cols; ++i) {
        (*mask_harris)[i] = (*harris_resp)[i] > new_tresh ? true : false;
    }
    for (int i = 0; i < detect_mask->rows * detect_mask->cols; i++) {
        if (!((*detect_mask)[i] && (*mask_harris)[i]))
            (*detect_mask)[i] = false;
    }


    // 2.3 Non-maximal suppression
    // dil is an image where each local maxima value is propagated to its neighborhood (display it!)
    matrix<number> *kernel = getStructuringElement<number>(min_distance, min_distance);
    matrix<number> *dil = dilate(harris_resp, kernel);

    // we want to keep only elements which are local maximas in their neighborhood

    matrix<bool> *harris_resp_dil = new matrix<bool>(harris_resp->rows, harris_resp->cols);
    for (int i = 0; i < harris_resp->rows * harris_resp->cols; ++i) {
        (*harris_resp_dil)[i] = (*harris_resp)[i] == (*dil)[i] ? true : false; // keep only local maximas by comparing dil and harris_resp
    }
    for (int i = 0; i < harris_resp_dil->rows * harris_resp_dil->cols; ++i) {
        if (!((*detect_mask)[i] && (*harris_resp_dil)[i]))
            (*detect_mask)[i] = false;
    }

    // 3. Select, sort and filter candidates
    // get coordinates of candidates
    int cols = 0;
    for (int i = 0; i < detect_mask->rows * detect_mask->cols; ++i) {
        if ((*detect_mask)[i] == true)
            cols++;
    }
    matrix<int> *non_zero_indices = new matrix<int>(2, cols);
    auto candidates_coords = non_zero_indices->transpose();
    // ...and their values
    matrix<uint8_t> *candidates_values = new matrix<uint8_t>(1, cols);
    for (int i,j = 0; i < harris_resp->rows * harris_resp->cols; ++i) {
        if ((*detect_mask)[i])
            (*candidates_values)[j++] = (*harris_resp)[i];
    }

    // sort candidates
    matrix<int> *sorted_indices = new matrix<int>(1, cols);
    for (int i = 0; i < sorted_indices->rows * sorted_indices->cols; ++i) {
        (*sorted_indices)[i] = i;
    }
    quickSort<number>(sorted_indices, candidates_values, 0, cols - 1);
    // keep only the bests
    if (max_keypoints > cols)
        max_keypoints = cols;

    matrix<int> *best_corners_coordinates = new matrix<int>(max_keypoints, 2);
    for (int i = 0; i < max_keypoints; ++i) {
        (*best_corners_coordinates)[i] = (*candidates_coords)[(*sorted_indices)[i]];
    }

    return best_corners_coordinates;
}

#endif /* MORPH_HH */