#include <stdlib.h>
#include <cmath> 
#include "derivatices.hh"

using namespace std;

float* mgridY(int size) {
    int border = size+1;
    float* meshGrid = (float *) malloc((border) * (border) * sizeof(float));
    
    for (int i = -size; i < border; i++) {
        for (int j = -size; j < border; j ++) {
            meshGrid[i * border + j] = i;
        }
    } 
    return meshGrid;
}

float* mgridX(int size) {
    int border = size+1;
    float* meshGrid = (float *) malloc((border) * (border) * sizeof(float));
    
    for (int i = -size; i < border; i++) {
        for (int j = -size; j < border; j ++) {
            meshGrid[i * border + j] = j;
        }
    } 
    return meshGrid;
}

float* gauss_kernel(int size) {
    float* yGrid = mgridY(size);
    float* xGrid = mgridX(size); 

    float sigma = (float)size / 3; 

    float* meshGrid = (float *) malloc(pow(size * 2 + 1, 2) * sizeof(float));

    for (int i = 0; i < (size * 2) + 1; i ++) {
        for (int j = 0; j < (size * 2) + 1; j ++) {
            int index = i * (size * 2 + 1) + j;
            float x = xGrid[index];
            float y = yGrid[index];
            meshGrid[index] = exp(-(pow(x, 2) / 2 * pow(sigma, 2)) + pow(y, 2) / 2 * pow(sigma, 2)); 
        }
    }

    free(yGrid); 
    free(xGrid);

    return meshGrid; 
}   


tuple_array gauss_derivative_kernels(int size) {
    float* yGrid = mgridY(size);
    float* xGrid = mgridX(size); 


    float* gx = (float *) malloc(pow(size * 2 + 1, 2) * sizeof(float)); 
    float* gy = (float *) malloc(pow(size * 2 + 1, 2) * sizeof(float));

    float sigma = (float)size / 3;

    for (int i = 0; i < (size * 2) + 1; i ++) {
        for (int j = 0; j < (size * 2) + 1; j ++) {
            int index = i * (size * 2 + 1) + j;
            float x = xGrid[index];
            float y = yGrid[index];
            float val = exp(-(pow(x, 2) / 2 * pow(sigma, 2)) + pow(y, 2) / 2 * pow(sigma, 2));
            gx[index] = -x * val; 
            gy[index] = -y * val;
        }
    }

    free(yGrid); 
    free(xGrid);

    tuple_array res {
        gx, 
        gy,
    };
    
    return res;
}