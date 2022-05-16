#include "harris.hh"
#include "convolve.hh"
#include "morph.hh"

#include <chrono>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void set_array_bool(bool *mat, int mat_rows, int mat_cols, bool value) {
        
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    mat[i * mat_cols + j] = value;
}

__global__ void set_bool_inferior(bool* output, float *mat, int mat_rows, int mat_cols, float threshold) {
        
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    output[i * mat_cols + j] = mat[i * mat_cols + j] > threshold;
}

__global__ void set_bool_equal(bool*output, float* mat1, float*mat2, int mat_rows, int mat_cols) {
        
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    output[i * mat_cols + j] = mat1[i * mat_cols + j] == mat1[i * mat_cols + j];
}

__global__ void set_bool_inverse(bool* m1, bool *m2, int mat_rows, int mat_cols) {
        
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (!(m1[i * mat_cols + j] && m2[i * mat_cols + j]))
        m1[i * mat_cols + j] = false;
}

matrix<bool>* create_array_bool(int mat_rows, int mat_cols, bool value) {
    bool* output;

    cudaMallocManaged(&output, mat_rows * mat_cols * sizeof(bool));
    gpuErrchk(cudaGetLastError());

    matrix<bool> *detect_mask = new matrix<bool>(mat_rows, mat_cols, output);
    
    int tx = 24;
    int ty = 16;

    dim3 blocks(mat_cols / tx, mat_rows / ty);
    dim3 threads(tx, ty);
    
    set_array_bool<<<blocks, threads>>>(output, mat_rows, mat_cols, true);
    gpuErrchk(cudaGetLastError());


    return detect_mask;
}

matrix<bool> *create_mask_harris(int mat_rows, int mat_cols, float threshold, float* harris_resp) {
    bool* output;

    cudaMalloc(&output, mat_rows * mat_cols * sizeof(bool));
    gpuErrchk(cudaGetLastError());

    matrix<bool> *mask = new matrix<bool>(mat_rows, mat_cols, output);
    
    int tx = 24;
    int ty = 16;

    dim3 blocks(mat_cols / tx, mat_rows / ty);
    dim3 threads(tx, ty);
    
    set_bool_inferior<<<blocks, threads>>>(output, harris_resp, mat_rows, mat_cols, threshold);
    gpuErrchk(cudaGetLastError());

    return mask;
}

void matrix_compare_inverse(bool* m1, bool* m2, int mat_rows, int mat_cols) {
    int tx = 24;
    int ty = 16;

    dim3 blocks(mat_cols / tx, mat_rows / ty);
    dim3 threads(tx, ty);

    set_bool_inverse<<<blocks, threads>>>(m1, m2, mat_rows, mat_cols);
    gpuErrchk(cudaGetLastError());
}

matrix<bool>* matrix_compare_equal(float* m1, float* m2, int mat_rows, int mat_cols) {
    bool* output;

    cudaMalloc(&output, mat_rows * mat_cols * sizeof(bool));
    gpuErrchk(cudaGetLastError());

    matrix<bool> *mask = new matrix<bool>(mat_rows, mat_cols, output);
    
    int tx = 24;
    int ty = 16;

    dim3 blocks(mat_cols / tx, mat_rows / ty);
    dim3 threads(tx, ty);

    set_bool_equal<<<blocks, threads>>>(output, m1, m2, mat_rows, mat_cols);
    gpuErrchk(cudaGetLastError());

    return mask;
}

matrix<float> *compute_harris_response(matrix<uint8_t> *img) {
  int derivativeKernelSize = 1;
  int opening_size = 1;

  tuple_matrix<float> tupleImxy =
      gauss_derivatives(img, derivativeKernelSize);

  matrix<float> *gauss = gauss_kernel(opening_size);

  matrix<float> *imxx =
      mat_multiply_element_wise(tupleImxy.mat1, tupleImxy.mat1);
  matrix<float> *imyy =
      mat_multiply_element_wise(tupleImxy.mat2, tupleImxy.mat2);
  matrix<float> *imxy =
      mat_multiply_element_wise(tupleImxy.mat1, tupleImxy.mat2);

  float *gauss_gpu;
  cudaMalloc((void **) &gauss_gpu, gauss->rows * gauss->cols * sizeof(float));

  cudaMemcpy(gauss_gpu, gauss->values, gauss->rows * gauss->cols * sizeof(float), cudaMemcpyHostToDevice);
  gpuErrchk(cudaGetLastError());

  matrix<float> *wxx = convolve(imxx, gauss_gpu, gauss->rows, gauss->cols);
  matrix<float> *wxy = convolve(imxy, gauss_gpu, gauss->rows, gauss->cols);
  matrix<float> *wyy = convolve(imyy, gauss_gpu, gauss->rows, gauss->cols);

  matrix<float> *wxxwyy = mat_multiply_element_wise(wxx, wyy);
  matrix<float> *wxyxy = mat_multiply_element_wise(wxy, wxy);

  matrix<float> *wdet = mat_diff_element_wise(wxxwyy, wxyxy);

  matrix<float> *wtr = mat_add_element_wise(wxx, wyy);

  matrix<float> *wtr1 =  *wtr + 1;
  
  matrix<float> *res = mat_divide_element_wise(wdet, wtr1);

  cudaFree(tupleImxy.mat1->values);
  cudaFree(tupleImxy.mat2->values);
  delete gauss;
  cudaFree(gauss_gpu);
  cudaFree(imxx->values);
  cudaFree(imyy->values);
  cudaFree(imxy->values);
  cudaFree(wxx->values);
  cudaFree(wyy->values);
  cudaFree(wxy->values);
  cudaFree(wxxwyy->values);
  cudaFree(wxyxy->values);
  delete wdet;
  delete wtr;
  delete wtr1;

  return res;
}

matrix<int> *detect_harris_points(matrix<uint8_t> *image_gray, int max_keypoints,
                                     int min_distance, float threshold)
{
    // 1. Compute Harris corner response
    auto time1 = std::chrono::system_clock::now();

    matrix<float> *harris_resp = compute_harris_response(image_gray);

    auto time2 = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = time2 - time1;
    std::cout << "Compute Harris corner response: " << diff.count() << "s" << std::endl;

    float* harris_resp_cu; 
    cudaMalloc(&harris_resp_cu, harris_resp->cols * harris_resp -> rows * sizeof(float));
    gpuErrchk(cudaGetLastError());

    cudaMemcpy(harris_resp_cu, harris_resp->values, harris_resp->cols * harris_resp -> rows * sizeof(float), cudaMemcpyHostToDevice);
    gpuErrchk(cudaGetLastError());
    // 2. Filtering
    // 2.0 Mask init: all our filtering is performed using a mask
    matrix<bool> *detect_mask = create_array_bool(harris_resp->rows, harris_resp->cols, true);

    
    time1 = std::chrono::system_clock::now();
    diff = time1 - time2;
    std::cout << "Filtering: " << diff.count() << "s" << std::endl;

    // 2.2 Response threshold
    float min_harris_resp = harris_resp->min();
    float new_tresh = min_harris_resp + threshold * (harris_resp->max() - min_harris_resp);

    // remove low response elements
    matrix<bool> *mask_harris = create_mask_harris(harris_resp->rows, harris_resp->cols, new_tresh, harris_resp_cu);    
    matrix_compare_inverse(detect_mask->values, mask_harris->values, harris_resp->rows, harris_resp->cols);

    
    time2 = std::chrono::system_clock::now();
    diff = time2 - time1;
    std::cout << "Response threshold: " << diff.count() << "s" << std::endl;

    // 2.3 Non-maximal suppression
    // dil is an image where each local maxima value is propagated to its neighborhood (display it!)
    matrix<bool> *kernel = getStructuringElement(min_distance, min_distance);
    matrix<float> *dil = dilate(harris_resp, kernel);

    // we want to keep only elements which are local maximas in their neighborhood
    matrix<bool> *harris_resp_dil = matrix_compare_equal(dil->values, harris_resp_cu, harris_resp->rows, harris_resp->cols);
    matrix_compare_inverse(detect_mask->values, harris_resp_dil->values, harris_resp->rows, harris_resp->cols);
    gpuErrchk(cudaDeviceSynchronize());

    time1 = std::chrono::system_clock::now();
    diff = time1 - time2;
    std::cout << "Non-maximal suppression: " << diff.count() << "s" << std::endl;

    // 3. Select, sort and filter candidates

    // get coordinates of candidates
    matrix<int> *candidates_coords = detect_mask->non_zero_transposed();

    int nb_candidates = candidates_coords->rows;
    
    // ...and their values
    matrix<float> *candidates_values = new matrix<float>(1, nb_candidates);
    for (int i = 0, j = 0; i < harris_resp->rows * harris_resp->cols; ++i) {
        if ((*detect_mask)[i])
            (*candidates_values)[j++] = (*harris_resp)[i];
    }

    // sort candidates
    int *sorted_indices = (int *) malloc(nb_candidates * sizeof(int));
    for (int i = 0; i < nb_candidates; ++i) {
        sorted_indices[i] = i;
    }

    float *test_values = (float *) malloc(nb_candidates * sizeof(float));
    for (int i = 0; i < nb_candidates; ++i) {
        test_values[i] = (*candidates_values)[i];
    }

    thrust::sort_by_key(thrust::host, test_values, test_values + nb_candidates, sorted_indices);
    // keep only the bests

    if (max_keypoints > nb_candidates)
        max_keypoints = nb_candidates;

    matrix<int> *best_corners_coordinates = new matrix<int>(max_keypoints, 2);
    for (int i = 0; i < max_keypoints; ++i) {
        (*best_corners_coordinates)[i * 2] = (*candidates_coords)[sorted_indices[i] * 2];
        (*best_corners_coordinates)[i * 2 + 1] = (*candidates_coords)[sorted_indices[i] * 2 + 1];
    }

    time2 = std::chrono::system_clock::now();
    diff = time2 - time1;
    std::cout << "Select, sort and filter candidates: " << diff.count() << "s" << std::endl;

    delete harris_resp;
    cudaFree(harris_resp_cu);
    cudaFree(detect_mask->values);
    cudaFree(mask_harris->values);
    cudaFree(kernel->values);
    cudaFree(dil->values);
    cudaFree(harris_resp_dil->values);
    delete candidates_coords;
    delete candidates_values;
    free(sorted_indices);

    return best_corners_coordinates;
}