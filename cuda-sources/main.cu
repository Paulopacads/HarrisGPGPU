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
        fprintf(stderr, "Invalid parameters\nformat: ./cuHarrisDetector <file_path> <max_keypoints> [-print]\n");
        return 1;
    }

    const char *filename = argv[1];

    uint8_t *pixels = nullptr;
    int width, height, bpp;

    if (!stbi_info(filename, &width, &height, &bpp))
        printf("error: can't open image\n");
    
    pixels = stbi_load(filename, &width, &height, &bpp, 1);
    matrix<uint8_t> *gray_image = new matrix<uint8_t>(height, width, pixels);

    matrix<int> *response = detect_harris_points(gray_image, atoi(argv[2]));

    delete gray_image;

    if (argc == 4)
    {
        pixels = stbi_load(filename, &width, &height, &bpp, 3);
        matrix<uint8_t> *image = new matrix<uint8_t>(height, width * 3, pixels);

        paint(image, response);

        stbi_write_png("output.jpg", width, height, 3, pixels, width * 3);

        delete image;
    }
    delete response;

    return 0;
}