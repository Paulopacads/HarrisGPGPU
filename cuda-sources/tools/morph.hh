#pragma once

#include <cmath>
#include <stdexcept>
#include "uchar.h"

#include "../construct/matrix.hh"

matrix<float> *dilate(matrix<float> *m1, matrix<bool> *m2);

matrix<bool> *getStructuringElement(int rows, int cols);