import numpy as np
import matplotlib.pyplot as plt
import cv2

def improved_ripple_transfer():
    # --- 1. Load Images ---
    # Load source ripple image (Grayscale)
    img_ripple_src = cv2.imread('3.png', cv2.IMREAD_GRAYSCALE)
    # Load target background (Color)
    img_bg_color = cv2.imread('4.png', cv2.IMREAD_COLOR)

    if img_ripple_src is None or img_bg_color is None:
        print("Error: Images '3.png' or '4.png' not found.")
        return

    # --- 2. Resize Source to Match Target ---
    rows, cols, _ = img_bg_color.shape
    img_ripple_resized = cv2.resize(img_ripple_src, (cols, rows))

    # --- 3. Frequency Domain Transformation (FFT) ---
    # Convert image to float for FFT
    dft = np.fft.fft2(img_ripple_resized.astype(float))
    dft_shift = np.fft.fftshift(dft)

    # --- VISUALIZATION TASK: Create Magnitude Spectrum ---
    # Calculate magnitude spectrum (log scale) to visualize the frequency domain
    # Formula: 20 * log(1 + |F(u,v)|)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    # Normalize the spectrum to 0-255 for display/saving
    magnitude_spectrum_view = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_spectrum_view = magnitude_spectrum_view.astype(np.uint8)

    # --- 4. Create Gaussian High-Pass Filter ---
    # Instead of a hard circle (which causes ringing artifacts), we use a Gaussian filter.
    # This creates smoother, more realistic water ripples.
    crow, ccol = rows // 2, cols // 2
    
    # Create a grid of coordinates
    y, x = np.ogrid[:rows, :cols]
    # Calculate distance from center
    distance_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
    
    # Gaussian Filter Formula: 1 - exp(-D^2 / (2 * sigma^2))
    # sigma controls the cutoff frequency. 
    # Higher sigma = cuts more low frequencies (removing lighting/shadows).
    sigma = 30 
    gaussian_mask = 1 - np.exp(-(distance_from_center**2) / (2 * (sigma**2)))

    # Apply the mask to the shifted FFT
    fshift_filtered = dft_shift * gaussian_mask

    # --- 5. Inverse FFT (Return to Image Space) ---
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    # --- 6. Post-Processing the Ripples ---
    # Apply a slight blur to remove high-frequency digital noise (scan lines)
    # This makes the water look liquid, not pixelated.
    img_back_smooth = cv2.GaussianBlur(img_back, (3, 3), 0)

    # Enhance contrast (Gain)
    gain = 2.5
    ripple_layer = img_back_smooth * gain

    # --- 7. Blending with Green Background ---
    img_bg_float = img_bg_color.astype(np.float32)
    final_image = np.zeros_like(img_bg_float)

    # Add ripples to all channels
    for i in range(3):
        final_image[:, :, i] = img_bg_float[:, :, i] + ripple_layer

    # Clip to valid pixel range
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    # --- 8. Display Results ---
    plt.figure(figsize=(16, 10))

    # 1. Source Ripple Image
    plt.subplot(2, 3, 1)
    plt.imshow(img_ripple_resized, cmap='gray')
    plt.title('1. Original Water Source')
    plt.axis('off')

    # 2. Frequency Domain (Magnitude Spectrum)
    plt.subplot(2, 3, 2)
    plt.imshow(magnitude_spectrum_view, cmap='gray')
    plt.title('2. Frequency Domain (Magnitude Spectrum)')
    plt.axis('off')

    # 3. Filter Mask (Gaussian)
    plt.subplot(2, 3, 3)
    plt.imshow(gaussian_mask, cmap='gray')
    plt.title('3. Gaussian High-Pass Filter')
    plt.axis('off')

    # 4. Extracted Ripples (Pure Signal)
    plt.subplot(2, 3, 4)
    plt.imshow(img_back_smooth, cmap='gray')
    plt.title('4. Extracted Ripples (Cleaned)')
    plt.axis('off')

    # 5. Original Green Background
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(img_bg_color, cv2.COLOR_BGR2RGB))
    plt.title('5. Target Background')
    plt.axis('off')

    # 6. Final Result
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    plt.title('6. Final: Ripples on Green')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --- 9. Save Files ---
    cv2.imwrite('FFT_Spectrum.png', magnitude_spectrum_view)
    cv2.imwrite('Final_Realistic_Ripple.png', final_image)
    print("Images saved: 'FFT_Spectrum.png' and 'Final_Realistic_Ripple.png'")

if __name__ == "__main__":
    improved_ripple_transfer()