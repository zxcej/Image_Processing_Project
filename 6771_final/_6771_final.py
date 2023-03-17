import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2


def bilateral_filter(image, kernel_size, sigma_intensity, sigma_range):
    half_size = kernel_size // 2  
    height, width = image.shape[0], image.shape[1]
    RGB = 1
    dim = len(image.shape)
    if dim != 2:
        RGB = image.shape[2]
    #print(RGB)
    image = image.reshape(height, width, RGB)
    output_image = np.zeros(image.shape)
    

    for i in range(half_size, height - half_size):
        for j in range(half_size, width - half_size):
            for k in range(RGB):
                total_weight = 0.0
                total_feature = 0.0
                for x in range(-half_size, half_size+1):
                    for y in range(-half_size, half_size+1):
                        #Range weight (Gaussian)
                        weight_range = -(x ** 2 + y ** 2) / (2 * (sigma_range ** 2))
                        #Intensity difference (Also Gaussian)
                        intensity_feature = -(int(image[i][j][k]) - int(image[i + x][j + y][k])) ** 2 / (2 * (sigma_intensity ** 2))
                        weight = np.exp(weight_range + intensity_feature)
                        total_weight += weight
                        total_feature += (weight * image[i + x][j + y][k])
                        #Normalization
                curr_scale = total_feature / total_weight
                output_image[i][j][k] = curr_scale
    return output_image.astype(np.uint8)

def arguments(args: list) -> tuple:
    filename = args[1] if args[1:] else "random.png"
    sigma_intensity = float(args[2]) if args[2:] else 100.0
    sigma_range = float(args[3]) if args[3:] else 3.0
    if args[4:]:
        k_s = int(args[4])
        #Kernel size must be an odd number
        if k_s % 2 == 0:
            k_s += 1
    else:
        k_s = 9
    return filename, sigma_intensity, sigma_range, k_s


if __name__ == '__main__':
    file_path, sigma_intensity, sigma_range, kernel_size = arguments(sys.argv)
    file_path = 'cameraman.png'
    image = cv2.imread(file_path, 0)
    #print(image.shape)
    gauss = cv2.GaussianBlur(image, (9,9), 10)
    #bilateral = cv2.bilateralFilter(image, kernel_size, sigma_intensity, sigma_range)
    my_bilateral = bilateral_filter(image, 9, 10,10)
    #print(mat.shape)
    
    #cv2.imshow("cv2", bilateral)
    #cv2.imshow("My output", mat)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    figure = plt.figure(figsize=(10, 10))
    #plt.subplot(1, 3, 1), plt.title('a) Original image')
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off')
    #plt.subplot(2, 2, 2), plt.title('my gaussian filter')
    #plt.imshow(cv2.cvtColor(gauss, cv2.COLOR_BGR2RGB)), plt.axis('off')
    #plt.subplot(1, 3, 2), plt.title('b) Bilateral filter from OpenCV')
    #plt.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.subplot(1, 4, 1), plt.title('10')
    plt.imshow(cv2.cvtColor(my_bilateral, cv2.COLOR_BGR2RGB)), plt.axis('off')
    my_bilateral = bilateral_filter(image, 9,75,10)
    plt.subplot(1, 4, 2), plt.title('75')
    plt.imshow(cv2.cvtColor(my_bilateral, cv2.COLOR_BGR2RGB)), plt.axis('off')
    my_bilateral = bilateral_filter(image, 9,300,10)
    plt.subplot(1, 4, 3), plt.title('300')
    plt.imshow(cv2.cvtColor(my_bilateral, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.subplot(1, 4, 4), plt.title('Gaussian')
    plt.imshow(cv2.cvtColor(gauss, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.show()
    figure.savefig('result.png', bbox_inches = 'tight',
    pad_inches = 0)


