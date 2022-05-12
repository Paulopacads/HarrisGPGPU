#ifndef MATRIX_HH
#define MATRIX_HH

template <typename number>
class matrix {
    public:
        int rows;
        int cols;

        number *values;

        matrix(int rows, int cols);
        matrix(int rows, int cols, number *values);

        ~matrix();

        matrix<number> *operator*(float n);
        matrix<number> *operator/(float n);

        number &operator[](int i);
};

#endif /* MATRIX_HH */