#include "harris.hh"
#include "convolve.hh"
#include "morph.hh"

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

  matrix<float> *wdet = mat_diff_element_wise(
      mat_multiply_element_wise(wxx, wyy), mat_multiply_element_wise(wxy, wxy));

  matrix<float> *wtr = mat_add_element_wise(wxx, wyy);

  matrix<float> *wtr1 =  *wtr + 1;
  
  matrix<float> *res = mat_divide_element_wise(wdet, wtr1);

  /*
  delete gauss;
  delete imxx;
  delete imyy;
  delete imxy;
  delete wxx;
  delete wyy;
  delete wxy;
  delete wdet;
  delete wtr;

  return res;
  */
  return res;
}


matrix<int> *detect_harris_points(matrix<uint8_t> *image_gray, int max_keypoints,
                                     int min_distance, float threshold)
{
    // 1. Compute Harris corner response
    printf("Compute Harris corner response\n");

    matrix<float> *harris_resp = compute_harris_response(image_gray);

    harris_resp->view();

    // 2. Filtering
    printf("Filtering\n");
    // 2.0 Mask init: all our filtering is performed using a mask
    matrix<bool> *detect_mask = new matrix<bool>(harris_resp->rows, harris_resp->cols);
    for (int i = 0; i < detect_mask->rows * detect_mask->cols; i++) {
        (*detect_mask)[i] = true;
    }

    printf("detect_mask:\n");
    detect_mask->view();

    printf("bubble2maskeroded\n");

    // 2.1 Background and border removal
    matrix<bool> *mask_bubble = bubble2maskeroded(image_gray, min_distance);

    int count2 = 0;
    for (int i = 0; i < mask_bubble->rows * mask_bubble->cols; i++) {
        count2 += (*mask_bubble)[i];
    }
    printf("mask_bubble %d:\n", count2);
    mask_bubble->view();

    printf("mask_bubble\n");
    for (int i = 0; i < detect_mask->rows * detect_mask->cols; i++) {
        if (!((*detect_mask)[i] && (*mask_bubble)[i]))
            (*detect_mask)[i] = false;
    }

    // 2.2 Response threshold
    printf("Response threshold\n");


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

    printf("detect_mask:\n");
    detect_mask->view();

    // 2.3 Non-maximal suppression
    printf("Non-maximal suppression\n");

    // dil is an image where each local maxima value is propagated to its neighborhood (display it!)
    matrix<bool> *kernel = getStructuringElement(min_distance, min_distance);
    matrix<float> *dil = dilate(harris_resp, kernel);

    int count = 0;
    for (int i = 0; i < dil->rows * dil->cols; ++i) {
        if ((*dil)[i] != 0)
            count++;
    }

    printf("count: %d\n", count);

    // we want to keep only elements which are local maximas in their neighborhood
    matrix<bool> *harris_resp_dil = new matrix<bool>(harris_resp->rows, harris_resp->cols);
    for (int i = 0; i < harris_resp->rows * harris_resp->cols; ++i) {
        (*harris_resp_dil)[i] = (*harris_resp)[i] == (*dil)[i] ? true : false; // keep only local maximas by comparing dil and harris_resp
    }
    for (int i = 0; i < harris_resp_dil->rows * harris_resp_dil->cols; ++i) {
        if (!((*detect_mask)[i] && (*harris_resp_dil)[i]))
            (*detect_mask)[i] = false;
    }

    // 3. Select, sort and filter candidates
    printf("Select, sort and filter candidates\n");

    // get coordinates of candidates

    //matrix<uint8_t> *non_zero_indices = new matrix<uint8_t>(2, cols);
    matrix<int> *candidates_coords = detect_mask->non_zero_transposed();

    int nb_candidates = candidates_coords->rows;
    
    // ...and their values
    matrix<float> *candidates_values = new matrix<float>(1, nb_candidates);
    for (int i = 0, j = 0; i < harris_resp->rows * harris_resp->cols; ++i) {
        if ((*detect_mask)[i])
            (*candidates_values)[j++] = (*harris_resp)[i];
    }

    printf("sort candidates\n");

    // sort candidates
    matrix<int> *sorted_indices = new matrix<int>(1, nb_candidates);
    for (int i = 0; i < sorted_indices->rows * sorted_indices->cols; ++i) {
        (*sorted_indices)[i] = i;
    }
    quickSort(sorted_indices, candidates_values, 0, nb_candidates - 1);
    // keep only the bests
    printf("keep only the bests\n");

    if (max_keypoints > nb_candidates)
        max_keypoints = nb_candidates;

    matrix<int> *best_corners_coordinates = new matrix<int>(max_keypoints, 2);
    for (int i = 0; i < max_keypoints; ++i) {
        (*best_corners_coordinates)[i] = (*candidates_coords)[(*sorted_indices)[i]];
    }

    return best_corners_coordinates;
}