#include "convolve.hh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename num>
__global__ void convolve_pixel(float *output, num *m1, float *m2, int m1_rows, int m2_rows,
    int m1_cols, int m2_cols) {

    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    int x_pad = m2_rows / 2;
    int y_pad = m2_cols / 2;

    float conv = 0;

    for (int k = 0; k < m2_rows; k++) {
        for (int l = 0; l < m2_cols; l++) {
            float value = 0;
            if (m2[k * m2_cols + l] && i - x_pad + k >= 0
            && i - x_pad + k < m1_rows && j - y_pad + l >= 0
            && j - y_pad + l < m1_cols) {
                value = m1[(i - x_pad + k) * m1_cols + j - y_pad + l];
            }
            conv += m2[k * m2_cols + l] * value;
        }
    }
    output[i * m1_cols + j] = conv;
}

matrix<float> *convolve(uint8_t *m1, matrix<float> *m2, int m1_rows, int m1_cols) {
    int tx = 24;
    int ty = 16;

    dim3 blocks(m1_cols / tx, m1_rows / ty);
    dim3 threads(tx, ty);

    float *output_gpu;
    float *m2_gpu;

    cudaMallocManaged(&output_gpu,  m1_rows * m1_cols * sizeof(float));
    gpuErrchk(cudaGetLastError());

    cudaMalloc((void **) &m2_gpu, m2->rows * m2->cols * sizeof(float));

    matrix<float> *output = new matrix<float>(m1_rows, m1_cols, output_gpu);

    cudaMemcpy(m2_gpu, m2->values, m2->rows * m2->cols * sizeof(float), cudaMemcpyHostToDevice);
    gpuErrchk(cudaGetLastError());
    
    convolve_pixel<<<blocks, threads>>>(output->values, m1, m2_gpu,
    m1_rows, m2->rows, m1_cols, m2->cols);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFree(m2_gpu);

    return output;
}

matrix<float> *convolve(matrix<float> *m1, float *m2, int m2_rows, int m2_cols) {
    int tx = 24;
    int ty = 16;

    dim3 blocks(m1->cols / tx, m1->rows / ty);
    dim3 threads(tx, ty);

    float *output_gpu;
    float *m1_gpu;

    cudaMallocManaged(&output_gpu,  m1->rows * m1->cols * sizeof(float));
    gpuErrchk(cudaGetLastError());

    cudaMalloc((void **) &m1_gpu, m1->rows * m1->cols * sizeof(float));

    matrix<float> *output = new matrix<float>(m1->rows, m1->cols, output_gpu);

    cudaMemcpy(m1_gpu, m1->values, m1->rows * m1->cols * sizeof(float), cudaMemcpyHostToDevice);
    gpuErrchk(cudaGetLastError());
    
    convolve_pixel<<<blocks, threads>>>(output->values, m1_gpu, m2,
    m1->rows, m2_rows, m1->cols, m2_cols);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFree(m1_gpu);

    return output;
}

template void __global__ convolve_pixel<uint8_t>(float *output, uint8_t *m1, float *m2, int m1_rows, int m2_rows, int m1_cols, int m2_cols);
template void __global__ convolve_pixel<float>(float *output, float *m1, float *m2, int m1_rows, int m2_rows, int m1_cols, int m2_cols);