# Bilateral Filter

This is a Python code that implements the Bilateral Filter algorithm, which is a non-linear filter that preserves edges while removing noise from an image. It uses a weighted average of neighboring pixels, where the weights depend on the similarity of pixel intensities and spatial proximity. 

## How to use

To run the script, you need to have the following packages installed: `numpy`, `matplotlib`, and `cv2`. Once you have installed the dependencies, you can run the script by executing the following command on the terminal:

```bash
python bilateral_filter.py [filename] [sigma_intensity] [sigma_range] [kernel_size]
```

Where:
- `[filename]` is the path of the image you want to apply the filter to. If this argument is not provided, a random image is generated.
- `[sigma_intensity]` is the standard deviation of the Gaussian filter applied in the intensity domain. The default value is 100.0.
- `[sigma_range]` is the standard deviation of the Gaussian filter applied in the spatial domain. The default value is 3.0.
- `[kernel_size]` is the size of the kernel used in the filter. It must be an odd number. The default value is 9.

The script will generate an output image with the applied filter, and save it in the same directory with the name "result.png". It will also show a plot with four images: the original image, the output of the bilateral filter with three different values of sigma intensity, and the output of the Gaussian filter. 

Note: The script currently sets `file_path` to `'cameraman.png'`, so the image being filtered will always be the cameraman image regardless of what filename is provided as an argument. To use a different image, either modify the `file_path` assignment in the script or provide a filename as an argument.

## Report

The report for this project can be found [here]()


## References

- [Bilateral Filter](https://en.wikipedia.org/wiki/Bilateral_filter)
- [Bilateral Filter - OpenCV](https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html)