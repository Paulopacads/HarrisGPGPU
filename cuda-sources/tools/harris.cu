#include "harris.hh"
#include "convolve.hh"
#include "morph.hh"

#include <chrono>
#include <iostream>

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

  matrix<float> *wxx = convolve(imxx, gauss);
  matrix<float> *wxy = convolve(imxy, gauss);
  matrix<float> *wyy = convolve(imyy, gauss);

  matrix<float> *wxxwyy = mat_multiply_element_wise(wxx, wyy);
  matrix<float> *wxyxy = mat_multiply_element_wise(wxy, wxy);

  matrix<float> *wdet = mat_diff_element_wise(wxxwyy, wxyxy);

  matrix<float> *wtr = mat_add_element_wise(wxx, wyy);

  matrix<float> *wtr1 =  *wtr + 1;
  
  matrix<float> *res = mat_divide_element_wise(wdet, wtr1);

  delete tupleImxy.mat1;
  delete tupleImxy.mat2;
  delete gauss;
  delete imxx;
  delete imyy;
  delete imxy;
  delete wxx;
  delete wyy;
  delete wxy;
  delete wxxwyy;
  delete wxyxy;
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

    // 2. Filtering
    // 2.0 Mask init: all our filtering is performed using a mask
    matrix<bool> *detect_mask = new matrix<bool>(harris_resp->rows, harris_resp->cols);
    for (int i = 0; i < detect_mask->rows * detect_mask->cols; i++) {
        (*detect_mask)[i] = true;
    }

    time1 = std::chrono::system_clock::now();
    diff = time1 - time2;
    std::cout << "Filtering: " << diff.count() << "s" << std::endl;

    // 2.2 Response threshold
    uint8_t min_harris_resp = harris_resp->min();
    auto new_tresh = min_harris_resp + threshold * (harris_resp->max() - min_harris_resp);

    // remove low response elements
    matrix<bool> *mask_harris = new matrix<bool>(harris_resp->rows, harris_resp->cols);
    for (int i = 0; i < mask_harris->rows * mask_harris->cols; ++i) {
        (*mask_harris)[i] = (*harris_resp)[i] > new_tresh ? true : false;
    }

    for (int i = 0; i < detect_mask->rows * detect_mask->cols; i++) {
        if (!((*detect_mask)[i] && (*mask_harris)[i]))
            (*detect_mask)[i] = false;
    }

    time2 = std::chrono::system_clock::now();
    diff = time2 - time1;
    std::cout << "Response threshold: " << diff.count() << "s" << std::endl;

    // 2.3 Non-maximal suppression
    // dil is an image where each local maxima value is propagated to its neighborhood (display it!)
    matrix<bool> *kernel = getStructuringElement(min_distance, min_distance);
    matrix<float> *dil = dilate(harris_resp, kernel);

    // we want to keep only elements which are local maximas in their neighborhood
    matrix<bool> *harris_resp_dil = new matrix<bool>(harris_resp->rows, harris_resp->cols);
    for (int i = 0; i < harris_resp->rows * harris_resp->cols; ++i) {
        (*harris_resp_dil)[i] = (*harris_resp)[i] == (*dil)[i] ? true : false; // keep only local maximas by comparing dil and harris_resp
    }
    for (int i = 0; i < harris_resp_dil->rows * harris_resp_dil->cols; ++i) {
        if (!((*detect_mask)[i] && (*harris_resp_dil)[i]))
            (*detect_mask)[i] = false;
    }

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
    matrix<int> *sorted_indices = new matrix<int>(1, nb_candidates);
    for (int i = 0; i < sorted_indices->rows * sorted_indices->cols; ++i) {
        (*sorted_indices)[i] = i;
    }
    quickSort(sorted_indices, candidates_values, 0, nb_candidates - 1);
    // keep only the bests

    if (max_keypoints > nb_candidates)
        max_keypoints = nb_candidates;

    matrix<int> *best_corners_coordinates = new matrix<int>(max_keypoints, 2);
    for (int i = 0; i < max_keypoints; ++i) {
        (*best_corners_coordinates)[i * 2] = (*candidates_coords)[(*sorted_indices)[i] * 2];
        (*best_corners_coordinates)[i * 2 + 1] = (*candidates_coords)[(*sorted_indices)[i] * 2 + 1];
    }

    time2 = std::chrono::system_clock::now();
    diff = time2 - time1;
    std::cout << "Select, sort and filter candidates: " << diff.count() << "s" << std::endl;

    delete harris_resp;
    delete detect_mask;
    delete mask_harris;
    delete kernel;
    cudaFree(dil->values);
    delete harris_resp_dil;
    delete candidates_coords;
    delete candidates_values;
    delete sorted_indices;

    return best_corners_coordinates;
}