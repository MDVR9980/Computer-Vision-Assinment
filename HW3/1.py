import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# --- 1. Load and Pre-process the Image ---
# Load the input image (1.png) in grayscale.
img_clean = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

if img_clean is None:
    print("Error: Input file '1.png' not found.")
    exit()

# Normalize the image to float (0 to 1) for FFT calculations.
img_float = img_clean.astype(np.float64) / 255.0
rows, cols = img_float.shape
center_r, center_c = rows // 2, cols // 2

# --- 2. Perform FFT (Fast Fourier Transform) ---
# Convert image to the frequency domain.
f_transform = np.fft.fft2(img_float)
# Shift the zero-frequency component (DC) to the center.
f_shift = np.fft.fftshift(f_transform)

# --- 3. Introduce Periodic Noise (Diagonal Stripes) ---

# Frequency Configuration:
# Higher numbers = tighter/denser lines. 
# Values around 30-35 match your reference image well.
u_noise = 32  # Vertical frequency
v_noise = 32  # Horizontal frequency

# Calculate coordinates for the noise peaks.
# Placing peaks in opposite quadrants creates diagonal lines.
peak_r1, peak_c1 = center_r + u_noise, center_c + v_noise
peak_r2, peak_c2 = center_r - u_noise, center_c - v_noise

# Amplitude Configuration (Visibility Control):
# This controls how "strong" the stripes are versus the background.
# - Previous value (0.4): Stripes were opaque, background hidden.
# - New value (0.25): Strong stripes, but background is visible behind them.
# - To see MORE background: reduce to 0.15 or 0.1
noise_amplitude = (rows * cols) * 0.25

# Add the noise spikes to the frequency spectrum.
# We add the same value to symmetric points to ensure the result is real.
f_shift[peak_r1, peak_c1] += noise_amplitude
f_shift[peak_r2, peak_c2] += noise_amplitude

# --- 4. Perform Inverse FFT ---
# Shift the zero-frequency component back to the corner.
f_ishift = np.fft.ifftshift(f_shift)
# Inverse FFT to get back to the spatial domain.
img_noisy = np.fft.ifft2(f_ishift)
# Take the real part (discard imaginary residuals).
img_noisy = np.real(img_noisy)

# --- 5. Normalize and Save ---
# Clip values to stay within valid range [0, 1] and convert to 0-255.
img_noisy_final = np.clip(img_noisy, 0, 1) * 255
img_noisy_final = img_noisy_final.astype(np.uint8)

# --- Display Results ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_clean, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_noisy_final, cmap='gray')
plt.title('Result: Stripes with Visible Background')
plt.axis('off')

plt.tight_layout()
# plt.show() # Uncomment if running in an environment with a display

# Save the output
output_filename = 'FINAL_STRIPES_VISIBLE_BG.png'
cv2.imwrite(output_filename, img_noisy_final)
print(f"Process complete. Image saved as '{output_filename}'.")