#include "construct/matrix.hh"
#include "tools/convolve.hh"
#include "tools/harris.hh"
#include "tools/morph.hh"
#include "tools/paint.hh"

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#include <cstdio>
#include <cstdint>

int main(int argc, char **argv) {
    if (argc < 3 && argc > 4) {
        fprintf(stderr, "Invalid parameters\nformat: ./cppHarrisDetector <file_path> <max_keypoints> [-print]\n");
        return 1;
    }

    const char *filename = argv[1];

    int dim = 1;
    uint8_t *pixels = nullptr;
    int width, height, bpp;

    if (!stbi_info(filename, &width, &height, &bpp))
        printf("error: can't open image\n");
    
    pixels = stbi_load(filename, &width, &height, &bpp, dim);
    matrix<uint8_t> *image = new matrix<uint8_t>(height, width, pixels);

    matrix<int> *response = detect_harris_points(image, atoi(argv[2]));

    if (argc == 4)
    {
        paint(image, response);

        stbi_write_png("output.jpg", width, height, 1, pixels, width * 1);
    }

    delete image;
    delete response;

    return 0;
}