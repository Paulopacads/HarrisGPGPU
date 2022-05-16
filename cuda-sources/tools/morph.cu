#include "morph.hh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void dilate_pixel(float *output, float *m1, bool *m2, int m1_rows, int m2_rows,
    int m1_cols, int m2_cols) {
        
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    int x_pad = m2_rows / 2;
    int y_pad = m2_cols / 2;

    float max = 0;

    for (int k = 0; k < m2_rows; k++) {
        for (int l = 0; l < m2_cols; l++) {
            if (m2[k * m2_cols + l] && i - x_pad + k >= 0
            && i - x_pad + k < m1_rows && j - y_pad + l >= 0
            && j - y_pad + l < m1_cols) {
                float tmp = m1[(i - x_pad + k) * m1_cols + j - y_pad + l];
                if (tmp > max)
                    max = tmp;
            }
        }
    }
    output[i * m1_cols + j] = max;
}

matrix<float> *dilate(matrix<float> *m1, matrix<bool> *m2) {
    int tx = 24;
    int ty = 16;

    dim3 blocks(m1->cols / tx, m1->rows / ty);
    dim3 threads(tx, ty);

    float *output_gpu;
    float *m1_gpu;
    bool *m2_gpu;

    cudaMalloc(&output_gpu,  m1->rows * m1->cols * sizeof(float));
    gpuErrchk(cudaGetLastError());

    cudaMalloc((void **) &m1_gpu, m1->rows * m1->cols * sizeof(float));
    cudaMalloc((void **) &m2_gpu, m2->rows * m2->cols * sizeof(bool));

    matrix<float> *output = new matrix<float>(m1->rows, m1->cols, output_gpu);

    cudaMemcpy(m1_gpu, m1->values, m1->rows * m1->cols * sizeof(float), cudaMemcpyHostToDevice);
    gpuErrchk(cudaGetLastError());

    cudaMemcpy(m2_gpu, m2->values, m2->rows * m2->cols * sizeof(bool), cudaMemcpyHostToDevice);
    gpuErrchk(cudaGetLastError());
    
    dilate_pixel<<<blocks, threads>>>(output->values, m1_gpu, m2_gpu,
    m1->rows, m2->rows, m1->cols, m2->cols);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaFree(m1_gpu);
    cudaFree(m2_gpu);

    return output;
}

__global__ void getStructuringGPU(bool *output, int rows, int cols) {

    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    
    int r = rows / 2;
    int c = cols / 2;

    int j1 = 0;
    int j2 = 0;
    double dy = i - r;
    if(-r <= dy && dy <= r)
    {
        double tmp = 1 - dy * dy / (r * r);
        int dx = r ? c * std::sqrt(tmp) : 0;
        j1 = c - dx - 1;
        j2 = c + dx + 1;
    }

    output[i * cols + j] = j < j2 && j > j1;
}

matrix<bool> *getStructuringElement(int rows, int cols) {
    int tx = 24;
    int ty = 16;

    dim3 blocks(cols / tx, rows / ty);
    dim3 threads(tx, ty);

    bool *output_gpu;

    cudaMallocManaged(&output_gpu, rows * cols * sizeof(bool));
    gpuErrchk(cudaGetLastError());

    matrix<bool> *output = new matrix<bool>(rows, cols, output_gpu);
    
    getStructuringGPU<<<blocks, threads>>>(output->values, rows, cols);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return output;
}
