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


void paint_point(matrix<uint8_t> *image, bool memory[], int color, double radius,
    point center, point p) {
    if (p.x >= 0 && p.y >= 0 && p.x < image->cols && p.y < image->rows
        && dist(center, p) < radius && memory[p.y * image->cols + p.x] == 0) {
        (*image)[p.y * image->cols + p.x] = color;
        memory[p.y * image->cols + p.x] = 1;
        paint_point(image, memory, color, radius, center, {p.x+1, p.y});
        paint_point(image, memory, color, radius, center, {p.x, p.y+1});
        paint_point(image, memory, color, radius, center, {p.x-1, p.y});
        paint_point(image, memory, color, radius, center, {p.x, p.y-1});
    }
}

void paint(matrix<uint8_t> *image, matrix<int> *points) {
    int max_x = 0;
    int max_y = 0;
    bool *memory = (bool*) calloc(image->rows * image->cols, sizeof(bool));
    for (size_t i = 0; i < points->rows; i++) {
        point start = {(*points)[i * 2], (*points)[i * 2 + 1]};
        if (start.x > max_x)
            max_x = start.x;
        if (start.y > max_y)
            max_y = start.y;
        //printf("point(%d, %d)\n", start.x, start.y);
        paint_point(image, memory, 255, 10, start, start);
    }
    printf("max x: %d\nmax y: %d\n", max_x, max_y);
    printf("rows: %d\ncols%d\n", image->rows, image->cols);
    free(memory);
}