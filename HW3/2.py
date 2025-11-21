import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def transfer_ripples_via_fft():
    # --- 1. Load Images ---
    # Load the source ripple image (Water) in Grayscale
    # We only need intensity data for the waves.
    img_ripple_src = cv2.imread('3.png', cv2.IMREAD_GRAYSCALE)
    
    # Load the target background (Green) in Color
    img_bg_color = cv2.imread('4.png', cv2.IMREAD_COLOR)

    # Check if images were loaded successfully
    if img_ripple_src is None or img_bg_color is None:
        print("Error: Could not load '3.png' or '4.png'. Make sure files exist in the directory.")
        return

    # --- 2. Resize Source to Match Target ---
    # Get dimensions of the green background
    rows, cols, channels = img_bg_color.shape
    
    # Resize the water image to match the green image size perfectly
    # This is necessary for pixel-to-pixel blending later.
    img_ripple_resized = cv2.resize(img_ripple_src, (cols, rows))

    # --- 3. Frequency Domain Processing (FFT) ---
    # Convert the spatial image to the frequency domain using Fast Fourier Transform.
    # This allows us to separate "details" (waves) from "general brightness".
    dft = np.fft.fft2(img_ripple_resized)
    dft_shift = np.fft.fftshift(dft) # Shift zero frequency (DC component) to the center

    # --- 4. High-Pass Filter Construction ---
    # We want to KEEP the waves (high frequencies) and REMOVE the average lighting (low frequencies).
    crow, ccol = rows // 2, cols // 2
    
    # Create a mask initialized to 1 (pass everything)
    mask = np.ones((rows, cols), np.uint8)
    
    # Define the radius of the center circle to block
    # A radius of 30-50 is usually good to remove global gradients/lighting.
    r = 40 
    
    # Create a circular mask to block low frequencies in the center
    center_y, center_x = np.ogrid[:rows, :cols]
    mask_area = (center_y - crow)**2 + (center_x - ccol)**2 <= r**2
    mask[mask_area] = 0 # Set center frequencies to 0
    
    # --- 5. Apply Filter and Inverse Transform ---
    # Apply the High-Pass Filter to the frequency domain image
    fshift_filtered = dft_shift * mask
    
    # Shift back the origin
    f_ishift = np.fft.ifftshift(fshift_filtered)
    
    # Perform Inverse Fast Fourier Transform to get back to the spatial domain
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back) # Take the real part of the complex output

    # --- 6. Blending ---
    # img_back now contains the "pure waves" centered around 0.
    # Positive values represent wave peaks, negative values represent wave troughs.
    
    # Optional: Increase the strength of the ripples (Contrast)
    ripple_strength = 2.0
    ripple_layer = img_back * ripple_strength

    # Convert background to float to allow negative math operations
    img_bg_float = img_bg_color.astype(np.float32)
    
    # Create a container for the final image
    final_image = np.zeros_like(img_bg_float)
    
    # Add the ripple layer to all three color channels (B, G, R)
    # This modulates the brightness of the green background based on the wave height.
    final_image[:, :, 0] = img_bg_float[:, :, 0] + ripple_layer # Blue channel
    final_image[:, :, 1] = img_bg_float[:, :, 1] + ripple_layer # Green channel
    final_image[:, :, 2] = img_bg_float[:, :, 2] + ripple_layer # Red channel

    # Clip values to ensure they remain in valid [0, 255] range
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    # --- 7. Display and Save Results ---
    plt.figure(figsize=(12, 8))

    # Show original ripple source
    plt.subplot(2, 2, 1)
    plt.imshow(img_ripple_resized, cmap='gray')
    plt.title('Source Ripple Image (3.png)')
    plt.axis('off')

    # Show extracted ripples (High-Pass Filtered)
    plt.subplot(2, 2, 2)
    plt.imshow(img_back, cmap='gray')
    plt.title('Extracted Ripples (Freq Domain Filtered)')
    plt.axis('off')

    # Show original background
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img_bg_color, cv2.COLOR_BGR2RGB))
    plt.title('Original Green Background (4.png)')
    plt.axis('off')

    # Show Final Result
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title('Final: Ripples Applied to Background')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save the result to disk
    output_filename = 'final_ripple_result.png'
    cv2.imwrite(output_filename, final_image)
    print(f"Processing complete. Image saved as '{output_filename}'")

if __name__ == "__main__":
    transfer_ripples_via_fft()