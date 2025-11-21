import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 

# --- 1. Load and Pre-process the Image ---
# Load the input image (1.png) in grayscale.
img_clean = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

if img_clean is None:
    print("Error: Input file '1.png' not found. Please ensure the file is in the script directory.")
    exit()

# Normalize the image to float for FFT computation (range 0 to 1).
img_float = img_clean.astype(np.float64) / 255.0

# Get image dimensions and center coordinates.
rows, cols = img_float.shape
center_r, center_c = rows // 2, cols // 2

# --- 2. 2D Fast Fourier Transform (FFT) and Shifting ---
# Compute the 2D-FFT.
f_transform = np.fft.fft2(img_float)

# Shift the zero-frequency component (DC) to the center of the spectrum.
f_shift = np.fft.fftshift(f_transform)

# --- 3. Introduce Periodic Noise in the Frequency Domain ---
# Periodic noise is added by introducing symmetric peaks in the spectrum.

# Noise Frequencies (Relative coordinates):
# These values determine the angle and density of the diagonal stripes (u_noise=25, v_noise=25 gives ~45 degree lines).
u_noise = 25  # Vertical frequency component of the noise
v_noise = 25  # Horizontal frequency component of the noise

# Absolute coordinates of the symmetric noise peaks.
peak_r1, peak_c1 = center_r + u_noise, center_c + v_noise
peak_r2, peak_c2 = center_r - u_noise, center_c - v_noise

# FINAL ADJUSTMENT: Reducing amplitude from 8000 to 5000 for better visibility of the background image.
noise_amplitude = 5000 

# Add the noise peaks to the shifted spectrum.
# Add the first peak
f_shift[peak_r1, peak_c1] += noise_amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi))

# Add the second symmetric peak (Ensures real output after IFFT)
f_shift[peak_r2, peak_c2] += noise_amplitude * np.exp(1j * np.random.uniform(0, 2*np.pi))


# --- 4. Inverse Fourier Transform ---
# Inverse shift (undo the fftshift).
f_ishift = np.fft.ifftshift(f_shift)

# Compute the Inverse 2D-FFT to return to the spatial domain.
img_noisy = np.fft.ifft2(f_ishift)

# Extract the real part and normalize/convert back to 8-bit integer (0-255).
img_noisy = np.real(img_noisy)
img_noisy_final = np.clip(img_noisy, 0, 1) * 255
img_noisy_final = img_noisy_final.astype(np.uint8)

# --- 5. Display and Save Output ---
plt.figure(figsize=(10, 5))

# Display original clean image
plt.subplot(1, 2, 1)
plt.imshow(img_clean, cmap='gray')
plt.title('Original Clean Image (1.png)')
plt.axis('off')

# Display output noisy image
plt.subplot(1, 2, 2)
plt.imshow(img_noisy_final, cmap='gray')
plt.title('Output with Periodic Noise (Final)')
plt.axis('off')

plt.tight_layout()
# plt.show() # Commented out due to non-interactive environment

# Save the resulting noisy image
output_filename = 'FINAL_NOISY_OUTPUT_v3.png'
cv2.imwrite(output_filename, img_noisy_final)
print(f"Periodic noisy image saved as '{output_filename}'.")

# Verification Step: Check if the file actually exists
if os.path.exists(output_filename):
    print(f"File successfully created and found: {output_filename}")
else:
    print(f"ERROR: File was NOT found after cv2.imwrite. Check directory permissions.")