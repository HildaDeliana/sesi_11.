 import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from scipy.ndimage import sobel


def read_and_convert_image(image_path):
    image = iio.imread(image_path, pilmode="L") 
    return image


def sobel_edge_detection(image):
    sobel_x = sobel(image, axis=0) 
    sobel_y = sobel(image, axis=1)  
    edge_magnitude = np.hypot(sobel_x, sobel_y) 
    edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255  
    return edge_magnitude.astype(np.uint8)


def basic_thresholding(image, threshold):
    thresholded_image = np.where(image > threshold, 255, 0)  
    return thresholded_image.astype(np.uint8)


if __name__ == "__main__":
   
    image_path = "C:\pemandangan.jpeg" 

   
    original_image = read_and_convert_image(image_path)

   
    edge_image = sobel_edge_detection(original_image)

    
    threshold_value = 100  
    segmented_image = basic_thresholding(edge_image, threshold_value)

    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Edge Detection (Sobel)")
    plt.imshow(edge_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Segmented Image (Threshold={threshold_value})")
    plt.imshow(segmented_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
