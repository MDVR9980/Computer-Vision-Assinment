import numpy as np
import cv2
import os
import matplotlib
# Set backend to 'Agg' to prevent QSocketNotifier warning on Linux
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# --- 1. Load and Pre-process the Image ---
img_clean = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

if img_clean is None:
    print("Error: Input file '1.png' not found.")
    exit()

img_float = img_clean.astype(np.float64) / 255.0
rows, cols = img_float.shape
center_r, center_c = rows // 2, cols // 2

# --- 2. FFT ---
f_transform = np.fft.fft2(img_float)
f_shift = np.fft.fftshift(f_transform)

# --- 3. Periodic Noise (Stripes) ---
# Frequency
u_noise = 32
v_noise = 32

# Coordinates
peak_r1, peak_c1 = center_r + u_noise, center_c + v_noise
peak_r2, peak_c2 = center_r - u_noise, center_c - v_noise

# Amplitude: 0.25 allows the background to be visible.
noise_amplitude = (rows * cols) * 0.25

# Add Noise
f_shift[peak_r1, peak_c1] += noise_amplitude
f_shift[peak_r2, peak_c2] += noise_amplitude

# --- 4. Inverse FFT ---
f_ishift = np.fft.ifftshift(f_shift)
img_noisy = np.fft.ifft2(f_ishift)
img_noisy = np.real(img_noisy)

# --- 5. Save ---
img_noisy_final = np.clip(img_noisy, 0, 1) * 255
img_noisy_final = img_noisy_final.astype(np.uint8)

# Save output
output_filename = 'FINAL_STRIPES_VISIBLE_BG.png'
cv2.imwrite(output_filename, img_noisy_final)
print(f"Process complete. Image saved as '{output_filename}'.")