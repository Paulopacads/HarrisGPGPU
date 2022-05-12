#include "construct/matrix.hh"
#include "tools/convolve.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#include <cstdio>
#include <cstdint>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Invalid parameters\nformat: harris.py <file_path> <max_keypoints>\n");
        return 1;
    }

    const char *filename = argv[1];

    int dim = 1;
    uint8_t *pixels = nullptr;
    int width, height, bpp;

    if (!stbi_info(filename, &width, &height, &bpp))
        printf("error: can't open image\n");
    
    pixels = stbi_load(filename, &width, &height, &bpp, dim);
    matrix<uint8_t> *image = new matrix<uint8_t>(width, height, pixels);

    for (int i = 0; i < 20; i++) {
        printf("%d\n", (*image)[i]);
    }

    matrix<float> *gaussian_blur = new matrix<float>(3, 3);
    gaussian_blur->values[0] = 1;
    gaussian_blur->values[1] = 2;
    gaussian_blur->values[2] = 1;
    gaussian_blur->values[3] = 2;
    gaussian_blur->values[4] = 4;
    gaussian_blur->values[5] = 2;
    gaussian_blur->values[6] = 1;
    gaussian_blur->values[7] = 2;
    gaussian_blur->values[8] = 1;

    for (int i = 0; i < 9; i++) {
        printf("%f\n", gaussian_blur->values[i]);
    }

    matrix<float> *gaussian_blur_2 = *gaussian_blur / 16;

    for (int i = 0; i < 9; i++) {
        printf("%f\n", gaussian_blur_2->values[i]);
    }

    matrix<uint8_t> *output = convolve(image, gaussian_blur_2);

    for (int i = 0; i < 20; i++) {
        printf("%d\n", (*output)[i]);
    }

    stbi_write_png("input.jpg", width, height, dim, image->values, width * dim);
    stbi_write_png("output.jpg", width, height, dim, output->values, width * dim);
    
    delete image;
    delete gaussian_blur;
    delete gaussian_blur_2;
    delete output;

    return 0;
}