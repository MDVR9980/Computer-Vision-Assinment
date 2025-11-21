import matplotlib
# --- LINUX FIX: Force Matplotlib to use Tkinter backend ---
# This prevents the "Wayland" and "QSocketNotifier" errors.
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def main():
    # --- 1. Load Images ---
    print("Loading images...")
    # Source: Water Ripples (Grayscale for intensity)
    img_ripple_src = cv2.imread('3.png', cv2.IMREAD_GRAYSCALE)
    # Target: Green Background (Color)
    img_bg_color = cv2.imread('4.png', cv2.IMREAD_COLOR)

    if img_ripple_src is None or img_bg_color is None:
        print("Error: Could not find '3.png' or '4.png'.")
        return

    # --- 2. Resize Source to Match Target ---
    rows, cols, _ = img_bg_color.shape
    img_ripple_resized = cv2.resize(img_ripple_src, (cols, rows))

    # --- 3. FFT (Frequency Domain Transformation) ---
    # Convert to float
    dft = np.fft.fft2(img_ripple_resized.astype(float))
    dft_shift = np.fft.fftshift(dft)

    # --- 4. Generate & Save Magnitude Spectrum (Task Requirement) ---
    # We use log scale because the center values are huge compared to edges.
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    # Normalize to 0-255 for saving as an image
    spectrum_img = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    spectrum_img = spectrum_img.astype(np.uint8)
    
    # Save the Spectrum Image immediately
    cv2.imwrite('FFT_Spectrum_View.png', spectrum_img)
    print(">> Saved: FFT_Spectrum_View.png")

    # --- 5. Gaussian High-Pass Filter ---
    # Create a Gaussian mask to remove low frequencies (lighting/shadows)
    # and keep high frequencies (ripples/edges).
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    
    # Calculate distance from center
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Sigma controls how much "blur" or "lighting" is removed.
    sigma = 30 
    # High Pass Formula: 1 - Gaussian_Low_Pass
    gaussian_mask = 1 - np.exp(-(distance**2) / (2 * (sigma**2)))

    # Apply Filter
    fshift_filtered = dft_shift * gaussian_mask

    # --- 6. Inverse FFT (Back to Image) ---
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    # --- 7. Post-Processing & Blending ---
    # 1. Smooth slightly to remove digital noise
    img_back_smooth = cv2.GaussianBlur(img_back, (3, 3), 0)
    
    # 2. Increase contrast (Gain) to make ripples visible on green
    gain = 2.5
    ripple_layer = img_back_smooth * gain

    # 3. Add to Green Background
    img_bg_float = img_bg_color.astype(np.float32)
    final_image = np.zeros_like(img_bg_float)

    # Apply ripple offset to all channels (Blue, Green, Red)
    for i in range(3):
        final_image[:, :, i] = img_bg_float[:, :, i] + ripple_layer

    # Clip values to valid 0-255 range
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    # Save the Final Image
    cv2.imwrite('Final_Ripple_Output.png', final_image)
    print(">> Saved: Final_Ripple_Output.png")

    # --- 8. Visualization (Matplotlib) ---
    print("Displaying results... (Close the window to finish)")
    plt.figure(figsize=(14, 6))

    # Show Spectrum
    plt.subplot(1, 3, 1)
    plt.imshow(spectrum_img, cmap='gray')
    plt.title('Frequency Spectrum\n(Saved as FFT_Spectrum_View.png)')
    plt.axis('off')

    # Show Filtered Ripple Mask
    plt.subplot(1, 3, 2)
    plt.imshow(img_back_smooth, cmap='gray')
    plt.title('Extracted Ripples (High-Pass)')
    plt.axis('off')

    # Show Final Result
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title('Final: Ripples on Green\n(Saved as Final_Ripple_Output.png)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()