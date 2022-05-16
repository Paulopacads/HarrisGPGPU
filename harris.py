import numpy as np
import cv2
import sys
import os
from typing import Tuple
from scipy import signal

def gauss_kernel(size: int) -> np.array:
    """
    Returns a 2D Gaussian kernel for convolutions.
    
    Parameters
    ----------
    size: int
        Size of the kernel to build
    
    Returns
    -------
    kernel: np.array of shape (size, size) and dtype np.float32
        Resulting Gaussian kernel where kernel[i,j] = Gaussian(i, j, mu=(0,0), sigma=(size/3, size/3))
    """
    size = int(size)
    y, x = np.mgrid[-size:size+1, -size:size+1]

    # x and y coefficients of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    
    # FIXME this is a box filter, adapt it to be a Gaussian
    # (this would also work, but would lead to poorer performance)

    sigma = (size / 3, size / 3)

    g = np.exp(-((x**2) / (2 * sigma[0]**2) + (y**2) / (2 * sigma[1]**2)))
    
    return g

def gauss_derivative_kernels(size: int) -> Tuple[np.array, np.array]:
    """
    Returns two 2D Gaussian derivative kernels (x and y) for convolutions.
    
    Parameters
    ----------
    size: int
        Size of the kernels to build
    
    Returns
    -------
    (gx, gy): tupe of (np.array, np.array), each of shape (size, size) and dtype np.float32
        Resulting Gaussian kernels where kernel[i,j] = Gaussian_z(i, j, mu=(0,0), sigma=(size/3, size/3))
        where Gaussian_z is either the x or the y Gaussian derivative.
    """
    size = int(size)
    y, x = np.mgrid[-size:size+1, -size:size+1]

    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    gx = np.ones((size*2+1, size*2+1))  # FIXME
    gy = np.ones((size*2+1, size*2+1))  # FIXME

    sigma = (size / 3, size / 3)

    gx = -x * np.exp(-((x**2) / (2 * sigma[0]**2) + (y**2) / (2 * sigma[1]**2)))
    gy = -y * np.exp(-((x**2) / (2 * sigma[0]**2) + (y**2) / (2 * sigma[1]**2)))

    return gx,gy

def gauss_derivatives(im: np.array, size: int) -> Tuple[np.array, np.array]:
    """
    Returns x and y gaussian derivatives for a given image.
    
    Parameters
    ----------
    im: np.array of shape (rows, cols)
        Input image
    size: int
        Size of the kernels to use
    
    Returns
    -------
    (Ix, Iy): tupe of (np.array, np.array), each of shape (rows, cols)
        Derivatives (x and y) of the image computed using Gaussian derivatives (with kernel of size `size`).
    """
    gx,gy = gauss_derivative_kernels(size)

    imx = signal.convolve(im, gx, mode='same')
    imy = signal.convolve(im, gy, mode='same')

    return imx,imy

def compute_harris_response(image: np.array) -> np.array:  
    """
    Returns the Harris cornerness response of a given image.
    
    Parameters
    ----------
    im: np.array of shape (rows, cols)
        Input image
    
    Returns
    -------
    response: np.array of shape (rows, cols) and dtype np.float32
        Harris cornerness response image.
    """
    DERIVATIVE_KERNEL_SIZE = 1
    OPENING_SIZE = 1

    #derivatives
    imx,imy = gauss_derivatives(image, DERIVATIVE_KERNEL_SIZE)

    #kernel for weighted sum
    gauss = gauss_kernel(OPENING_SIZE) # opening param

    #compute components of the structure tensor
    
    Wxx = signal.convolve(imx*imx, gauss, mode='same')
    Wxy = signal.convolve(imx*imy, gauss, mode='same')
    Wyy = signal.convolve(imy*imy, gauss, mode='same')

    #determinant and trace
    Wdet = Wxx * Wyy - Wxy * Wxy
    Wtr = Wxx + Wyy

    # return Wdet - k * Wtr**2 # k is hard to tune
    # return Wdet / Wtr # we would need to filter NaNs
    return Wdet / (Wtr + 1)  # 1 seems to be a reasonable value for epsilon

def bubble2maskeroded(img_gray: np.array, border: int=10) -> np.array:
    """
    Returns the eroded mask of a given image, to remove pixels which are close to the border.
    
    Parameters
    ----------
    im: np.array of shape (rows, cols)
        Input image
    
    Returns
    -------
    mask: np.array of shape (rows, cols) and dtype bool
        Image mask.
    """
    if img_gray.ndim > 2:
        raise ValueError(
            """bubble2maskeroded: img_gray must be a grayscale image.
            The image you passed has %d dimensions instead of 2.
            Try to convert it to grayscale before passing it to bubble2maskeroded.
            """ % (img_gray.ndim, ))
    mask = img_gray > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border*2,border*2))
    mask_er = cv2.erode(mask.astype(np.uint8), 
                        kernel, 
                        borderType=cv2.BORDER_CONSTANT, 
                        borderValue=0)
    return mask_er > 0

def detect_harris_points(image_gray: np.array, max_keypoints: int=30, 
                         min_distance: int=25, threshold: float=0.1) -> np.array:
    """
    Detects and returns a sorted list of coordinates for each corner keypoint detected in an image.
    
    Parameters
    ----------
    image_gray: np.array
        Input image
    max_keypoints: int, default=30
        Number of keypoints to return, at most (we may have less keypoints)
    min_distance: int, default=25
        Minimum distance between two keypoints
    threshold: float, default=0.1
        For each keypoint k_i, we ensure that its response h_i will verify
        $h_i > min(response) + threshold * (max(reponse) - min(response))$
    
    Returns
    -------
    corner_coord: np.array of shape (N, 2) and dtype int
        Array of corner keypoint 2D coordinates, with N <= max_keypoints
    """
    # 1. Compute Harris corner response
    harris_resp = compute_harris_response(image_gray)
    
    # 2. Filtering
    # 2.0 Mask init: all our filtering is performed using a mask
    detect_mask = np.ones(harris_resp.shape, dtype=bool)

    # 2.2 Response threshold
    new_tresh = np.min(harris_resp) + threshold * (np.max(harris_resp) - np.min(harris_resp))
    detect_mask &= harris_resp > new_tresh# FIXME <------------------------  # remove low response elements
    # 2.3 Non-maximal suppression
    # dil is an image where each local maxima value is propagated to its neighborhood (display it!)
    dil = cv2.dilate(harris_resp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_distance, min_distance)))
    # we want to keep only elements which are local maximas in their neighborhood
    detect_mask &= harris_resp == dil  # FIXME <------------ # keep only local maximas by comparing dil and harris_resp
               
    # 3. Select, sort and filter candidates
    # get coordinates of candidates
    candidates_coords = np.transpose(detect_mask.nonzero())
    # ...and their values
    candidate_values = harris_resp[detect_mask]
    # sort candidates
    sorted_indices = np.argsort(candidate_values) # FIXME <----------------------
    # keep only the bests
    if max_keypoints > len(candidates_coords):
        max_keypoints = len(candidates_coords)

    best_corners_coordinates = candidates_coords[sorted_indices][:max_keypoints]  # FIXME <-----------------------

    return best_corners_coordinates

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Invalid parameters\nformat: harris.py <file_path> <max_keypoints>\n")
        sys.exit(1)

    filename = sys.argv[1]
    max_keypoints = int(sys.argv[2])
    
    if not os.path.exists(filename):
        sys.stderr.write("File does not exists\n")
        sys.exit(1)

    try:
        image = cv2.imread(filename)

    except:
        sys.stderr.write("Can't read image\n")
        sys.exit(1)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris_points = detect_harris_points(image_gray, max_keypoints)

    for point in harris_points:
        image = cv2.circle(image, (point[1], point[0]), radius=10, color=(0, 0, 255), thickness=-1)

    cv2.imwrite("output.png", image)

    print(len(harris_points))

if __name__ == "__main__":
    main()