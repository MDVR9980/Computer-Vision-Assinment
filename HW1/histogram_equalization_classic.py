"""
Computer Vision - Homework 1
Student: Mohammad Davood VahhabRajaee - 4041419041
Topic: Histogram Equalization (8-bit Grayscale Image)
Language: Python 3
Libraries: numpy, pillow, matplotlib
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def histogram_equalization_classic(img_array):
    """
    Standard histogram equalization (classic method)
    Input: 8-bit grayscale image as numpy array
    Output: Equalized image as numpy array
    """
    flat = img_array.flatten()

    # Compute histogram
    hist, bins = np.histogram(flat, bins=256, range=[0, 256])

    # Compute classical CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # scale to 0-255

    # Map original pixel values to equalized values
    equalized = np.interp(flat, bins[:-1], cdf_normalized)

    # Reshape back to original image shape
    equalized_img = equalized.reshape(img_array.shape).astype(np.uint8)
    return equalized_img


def demo_with_image_classic(image_path, save_path="sea_equalized.png"):
    # Load grayscale image
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # Apply classical histogram equalization
    equalized_img = histogram_equalization_classic(img_array)

    # Save the equalized image
    Image.fromarray(equalized_img).save(save_path)
    print(f"✅ Equalized image saved as: {save_path}")

    # Plot images and histograms
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img_array, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(equalized_img, cmap="gray")
    plt.title("Equalized Image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.hist(img_array.flatten(), bins=256, color="gray")
    plt.title("Original Histogram")

    plt.subplot(2, 2, 4)
    plt.hist(equalized_img.flatten(), bins=256, color="gray")
    plt.title("Equalized Histogram")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_with_image_classic("sea_original.png", "sea_equalized.png")
