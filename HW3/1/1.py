import numpy as np
import matplotlib
# Force Matplotlib to use 'TkAgg' backend for displaying windows on Linux
# If you get an error here, run: sudo apt-get install python3-tk
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
import cv2
import os

# --- 1. Load Image ---
img_clean = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

if img_clean is None:
    print("Error: '1.png' not found.")
    exit()

# Normalize
img_float = img_clean.astype(np.float64) / 255.0
rows, cols = img_float.shape
center_r, center_c = rows // 2, cols // 2

# --- 2. FFT (Frequency Domain) ---
f_transform = np.fft.fft2(img_float)
f_shift = np.fft.fftshift(f_transform)

# --- 3. Add Periodic Noise ---
# Settings for diagonal stripes
u_noise = 32
v_noise = 32
noise_amplitude = (rows * cols) * 0.25  # 0.25 allows background visibility

# Peak coordinates
p1 = (center_r + u_noise, center_c + v_noise)
p2 = (center_r - u_noise, center_c - v_noise)

# Add noise spikes
f_shift[p1] += noise_amplitude
f_shift[p2] += noise_amplitude

# --- 4. GENERATE SPECTRUM IMAGE (CRITICAL STEP) ---
# We calculate the Magnitude Spectrum to visualize the frequency domain.
# Logarithm is used because the dynamic range is very high.
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

# Normalize the spectrum to 0-255 to save it as a PNG image
spectrum_img = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
spectrum_img = spectrum_img.astype(np.uint8)

# --- 5. Inverse FFT ---
f_ishift = np.fft.ifftshift(f_shift)
img_noisy = np.fft.ifft2(f_ishift)
img_noisy = np.real(img_noisy)

# Clip and convert to uint8
img_noisy_final = np.clip(img_noisy, 0, 1) * 255
img_noisy_final = img_noisy_final.astype(np.uint8)

# --- 6. Display (Matplotlib) ---
plt.figure(figsize=(15, 6))

# Plot 1: Original
plt.subplot(1, 3, 1)
plt.imshow(img_clean, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plot 2: Frequency Spectrum (The requested visualization)
plt.subplot(1, 3, 2)
plt.imshow(spectrum_img, cmap='gray')
plt.title('Frequency Domain (Spectrum)\nLook for white dots!')
plt.axis('off')
# Draw circles around the noise points to highlight them in the plot
plt.plot(p1[1], p1[0], 'ro', markersize=5, fillstyle='none', markeredgewidth=2)
plt.plot(p2[1], p2[0], 'ro', markersize=5, fillstyle='none', markeredgewidth=2)

# Plot 3: Final Result
plt.subplot(1, 3, 3)
plt.imshow(img_noisy_final, cmap='gray')
plt.title('Final Result (Stripes + Image)')
plt.axis('off')

plt.tight_layout()

print("Displaying plot... (Close the window to finish script)")
plt.show() # This will pop up the window

# --- 7. Save Images ---
# Save the Frequency Spectrum
cv2.imwrite('Frequency_Spectrum.png', spectrum_img)
print("Saved: Frequency_Spectrum.png")

# Save the Final Result
cv2.imwrite('Final_Output_Stripes.png', img_noisy_final)
print("Saved: Final_Output_Stripes.png")