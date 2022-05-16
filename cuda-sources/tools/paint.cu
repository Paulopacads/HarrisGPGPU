#include "paint.hh"
#include "../construct/point.hh"

#include <cstdlib>
#include <cmath>

#include <cstdio>

double dist(point p1, point p2) {
    int x = p1.x - p2.x;
    int y = p1.y - p2.y;
    return std::sqrt(x*x + y*y);
}


void paint_point(matrix<uint8_t> *image, bool memory[], double radius,
    point center, point p) {
    if (p.x >= 0 && p.y >= 0 && p.x < (image->cols / 3) && p.y < image->rows
        && dist(center, p) < radius && memory[p.y * (image->cols / 3) + p.x] == 0) {
        (*image)[p.y * image->cols + p.x * 3] = 255;
        (*image)[p.y * image->cols + p.x * 3 + 1] = 0;
        (*image)[p.y * image->cols + p.x * 3 + 2] = 0;
        memory[p.y * (image->cols / 3) + p.x] = 1;
        paint_point(image, memory, radius, center, {p.x+1, p.y});
        paint_point(image, memory, radius, center, {p.x, p.y+1});
        paint_point(image, memory, radius, center, {p.x-1, p.y});
        paint_point(image, memory, radius, center, {p.x, p.y-1});
    }
}

void paint(matrix<uint8_t> *image, matrix<int> *points) {
    int max_x = 0;
    int max_y = 0;
    bool *memory = (bool*) calloc(image->rows * (image->cols / 3), sizeof(bool));
    for (size_t i = 0; i < points->rows; i++) {
        point start = {(*points)[i * 2 + 1], (*points)[i * 2]};
        if (start.x > max_x)
            max_x = start.x;
        if (start.y > max_y)
            max_y = start.y;

        paint_point(image, memory, 10, start, start);
    }
    free(memory);
}