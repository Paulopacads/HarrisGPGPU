#include "construct/matrix.hh"
#include "tools/morph.hh"

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

    matrix<bool> *mask = new matrix<bool>(3, 3);
    mask->values[0] = true;
    mask->values[1] = true;
    mask->values[2] = true;
    mask->values[3] = true;
    mask->values[4] = true;
    mask->values[5] = true;
    mask->values[6] = true;
    mask->values[7] = true;
    mask->values[8] = true;

    matrix<uint8_t> *output = dilate(image, mask);

    stbi_write_png("input.jpg", width, height, dim, image->values, width * dim);
    stbi_write_png("output.jpg", width, height, dim, output->values, width * dim);
    
    delete image;
    delete mask;
    delete output;

    return 0;
}