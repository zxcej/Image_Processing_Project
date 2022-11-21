
"""
Implementation of Bilateral filter
Inputs:
    img: A 2d image with values in between 0 and 1
    varS: variance in space dimension.
    varI: variance in range parameter.
    N: Kernel size(Must be an odd number)
Output:
    img:A 2d zero padded image with values in between 0 and 1
"""
import sys
import numpy as np
import math
import cv2

# 1d gaussian filter function
def gaussian_1d(img_sec: np.ndarray, var: float) -> np.ndarray:
    sigma = math.sqrt(var)
    return (1/(math.sqrt(2*math.pi)*sigma)) * np.exp(-(img_sec**2)/(2*sigma**2))

#Gaussian kernel generator using the 1d function above
def gaussian_kernel(k_s: int, spatial_var: float) -> np.ndarray:
    a= (k_s,k_s)
    square = np.zeros(a)
    x,y = square.shape
    for i in range(0,x):
        for j in range(0,y):
            #Distance from the center (Figure 3.34 from textbook)
            square[i, j] = math.sqrt((i - k_s // 2) ** 2 + (j - k_s // 2) ** 2)
    return gaussian_1d(square, spatial_var)

#Get a submatrix from the original matrix
def sub_matrix(img: np.ndarray, x: int, y: int, k_s: int) -> np.ndarray:
    value_size = k_s // 2
    return img[x - value_size : x + value_size + 1, y - value_size : y + value_size + 1]


def my_bilateral_filter(original: np.ndarray,spatial_var: float,range_var: float,k_s: int,) -> np.ndarray:
    kernel = gaussian_kernel(k_s, spatial_var)
    value_size = k_s//2;
    filtered_img = np.zeros(original.shape)
    normalized = original/255
    normalized = np.pad(normalized, value_size, mode="constant")
    normalized = normalized.astype(np.float64)
    size_x, size_y = original.shape
    x = size_x - value_size
    y = size_y - value_size
    for i in range(value_size, x):
        for j in range(value_size,y):
            current_window = sub_matrix(normalized, i, j, k_s)
            range_diff_window = current_window - current_window[value_size, value_size]
            subimg_range = gaussian_1d(range_diff_window, range_var)
            weights = np.multiply(kernel, subimg_range)
            featured_window = np.multiply(current_window, weights)
            featured_val = np.sum(featured_window) / np.sum(weights)
            filtered_img[i, j] = featured_val
    filtered_img = filtered_img * 255
    filtered_img = filtered_img.astype(np.uint8)
    return filtered_img


def arguments(args: list) -> tuple:
    filename = args[1] if args[1:] else "random.jpg"
    spatial_var = float(args[2]) if args[2:] else 1.0
    range_var = float(args[3]) if args[3:] else 1.0
    if args[4:]:
        k_s = int(args[4])
        #Kernel size must be an odd number
        if k_s % 2 == 0:
            k_s += 1
    else:
        k_s = 3
    return filename, spatial_var, range_var, k_s


if __name__ == "__main__":
    filename, spatial_var, range_var, k_s = arguments(sys.argv)
    img = cv2.imread(filename, 0)
    cv2.imshow("input image", img)
    out = my_bilateral_filter(img, spatial_var, range_var, k_s)
    #out = cv2.bilateralFilter(img, 15, 75, 75)
    cv2.imshow("output image", out)
    #cv2.imwrite(r'C:\Users\x_zhu202\source\repos\6771_final\6771_final\rossobw.png',img)
    #cv2.imwrite(r'C:\Users\x_zhu202\source\repos\6771_final\6771_final\rossosample.png',out)
    #gaus = cv2.GaussianBlur(img, (9,9),75)
    #cv2.imshow("gaussian image", gaus)
    cv2.waitKey(0)
    cv2.destroyAllWindows()