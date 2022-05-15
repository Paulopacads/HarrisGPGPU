#include "morph.hh"

matrix<float> *dilate(matrix<float> *m1, matrix<bool> *m2) {
    matrix<float> *output = new matrix<float>(m1->rows, m1->cols);
    int x_pad = m2->rows / 2;
    int y_pad = m2->cols / 2;

    for (int i = 0; i < m1->rows; i++) {
        for (int j = 0; j < m1->cols; j++) {
            float max = 0;

            for (int k = 0; k < m2->rows; k++) {
                for (int l = 0; l < m2->cols; l++) {
                    if ((*m2)[k * m2->cols + l] && i - x_pad + k >= 0
                    && i - x_pad + k < m1->rows && j - y_pad + l >= 0
                    && j - y_pad + l < m1->cols) {
                        float tmp = (*m1)[(i - x_pad + k) * m1->cols + j - y_pad + l];
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

matrix<bool> *getStructuringElement(int rows, int cols)
{
    int i, j;
    int r = 0, c = 0;
    double inv_r2 = 0;

    r = rows/2;
    c = cols/2;
    inv_r2 = r ? 1./((double)r*r) : 0;

    matrix<bool> *elem = new matrix<bool>(rows, cols);

    for( i = 0; i < rows; i++ )
    {
        int j1 = 0, j2 = 0;
        int dy = i - r;
        if( std::abs(dy) <= r )
        {
            int dx = c*std::sqrt((r*r - dy*dy)*inv_r2);
            j1 = std::max( c - dx, 0 );
            j2 = std::min( c + dx + 1, cols );
        }

        for( j = 0; j < j1; j++ )
            (*elem)[i * cols + j] = false;
        for( ; j < j2; j++ )
            (*elem)[i * cols + j] = true;
        for( ; j < rows; j++ )
            (*elem)[i * cols + j] = false;
    }

    return elem;
}

matrix<bool> *bubble2maskeroded(matrix<uint8_t> *img_gray, int border)
{
    matrix<bool> *mask = *img_gray > 0;
    matrix<bool> *kernel = getStructuringElement(border*2, border*2);

    matrix<bool> *mask_er = erode(mask, kernel);
    return *mask_er > 0;
}