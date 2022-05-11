#ifndef MATRIX_HH
#define MATRIX_HH

class matrix {
    public:
        int rows;
        int cols;

        float *values;

        matrix(int rows, int cols);
        ~matrix();

        matrix *operator*(float n);

        float &operator[](int i);
};

#endif /* MATRIX_HH */