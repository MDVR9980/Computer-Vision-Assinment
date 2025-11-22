import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import os
import sys
import csv

# =========================================================
# 1. NOISE GENERATION FUNCTIONS (Task 1)
# =========================================================
def add_noise(image, noise_type, intensity=0.05, mean=0, var=0.01):
    """
    Adds specific noise types to an image based on the assignment requirements.
    """
    row, col = image.shape
    noisy_image = image.copy()
    
    # Salt and Pepper Noise
    if noise_type == "salt_pepper":
        amount = intensity
        s_vs_p = 0.5 # Salt vs Pepper ratio
        
        # Add Salt (White pixels)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 255
        
        # Add Pepper (Black pixels)
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
        
    # Salt Noise (White only)
    elif noise_type == "salt":
        num_salt = np.ceil(intensity * image.size)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 255

    # Pepper Noise (Black only)
    elif noise_type == "pepper":
        num_pepper = np.ceil(intensity * image.size)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
        
    # Gaussian Noise
    elif noise_type == "gaussian":
        # Adjust sigma based on variance and intensity
        sigma = (var + intensity) ** 0.5 
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        
        # Normalize to 0-1, add noise, clip, and convert back to 0-255
        noisy_image = noisy_image / 255.0 + gauss
        noisy_image = np.clip(noisy_image, 0, 1) * 255
        noisy_image = noisy_image.astype(np.uint8)

    # Uniform Noise
    elif noise_type == "uniform":
        noise = np.random.uniform(-intensity*255, intensity*255, (row, col))
        noisy_image = noisy_image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

# =========================================================
# 2. FILTERING FUNCTIONS (Task 3)
# =========================================================
def apply_filter(noisy_img, filter_type, kernel_size=3, Q=1.5, d=2):
    """
    Applies spatial filters to remove noise.
    """
    # Median Filter
    if filter_type == "median":
        return cv2.medianBlur(noisy_img, kernel_size)
    
    # Arithmetic Mean Filter
    elif filter_type == "mean": 
        return cv2.blur(noisy_img, (kernel_size, kernel_size))
    
    # Alpha-trimmed Mean Filter
    elif filter_type == "alpha_trimmed":
        def alpha_trimmed_mean_algo(buffer):
            sort_buf = np.sort(buffer)
            # Trim 'd' pixels from the beginning and end of the sorted array
            trim_slice = sort_buf[d:-d] if d > 0 else sort_buf
            return np.mean(trim_slice)
        
        return generic_filter(noisy_img, alpha_trimmed_mean_algo, size=kernel_size)

    # Contraharmonic Mean Filter
    elif filter_type == "contraharmonic":
        img_float = noisy_img.astype(np.float32)
        epsilon = 1e-5 # Small value to avoid division by zero
        
        # Calculate numerator: sum(pixel^(Q+1))
        numerator = cv2.boxFilter(np.power(img_float + epsilon, Q + 1), -1, (kernel_size, kernel_size), normalize=False)
        
        # Calculate denominator: sum(pixel^Q)
        denominator = cv2.boxFilter(np.power(img_float + epsilon, Q), -1, (kernel_size, kernel_size), normalize=False)
        
        result = numerator / (denominator + epsilon)
        return np.clip(result, 0, 255).astype(np.uint8)

    return noisy_img

# =========================================================
# 3. METRIC CALCULATION (Tasks 2, 4, 5)
# =========================================================
def calculate_metrics(f, g, k):
    """
    Calculates the specific metrics required by the assignment.
    Args:
        f: Original Image (Clean)
        g: Noisy Image
        k: Filtered/Restored Image
    """
    # Task 2: Count pixels affected by noise
    # Logic: f (original) is not equal to g (noisy)
    affected_count = np.sum(f != g)
    
    # Task 4: Count noisy pixels that were fully corrected to original state
    # Logic: (Pixel was noisy in g) AND (Pixel is now correct in k)
    corrected_count = np.sum((f != g) & (k == f))
    
    # Task 5: Count clean pixels that were corrupted/ruined by the filter
    # Logic: (Pixel was clean in g) BUT (Pixel is different in k)
    corrupted_count = np.sum((f == g) & (k != f))
    
    return affected_count, corrected_count, corrupted_count

# =========================================================
# 4. MAIN EXECUTION LOOP
# =========================================================
def main():
    # Input configuration
    target_image = 'Lenna.png'
    output_dir = "HW2_Results"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    # Check if input image exists
    if not os.path.exists(target_image):
        print(f"❌ Error: '{target_image}' not found.")
        print("Please ensure 'Lenna.png' is in the same directory as this script.")
        sys.exit(1)
    
    # Load image in Grayscale mode
    f = cv2.imread(target_image, cv2.IMREAD_GRAYSCALE)
    if f is None:
        print("❌ Error: Could not read image. Check file format.")
        sys.exit(1)

    print(f"✅ Processing image: {target_image} (Resolution: {f.shape})")
    
    # --- Experimental Parameters (Based on Assignment) ---
    noise_types = ["salt_pepper", "salt", "pepper", "gaussian", "uniform"]
    filter_types = ["median", "mean", "alpha_trimmed", "contraharmonic"]
    intensities = [0.05, 0.1, 0.2]  # Different intensities for analysis
    kernel_size = 3
    
    # Initialize CSV file to store numerical results
    csv_path = os.path.join(output_dir, "metrics_table.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write CSV Header
    csv_writer.writerow(["Noise Type", "Intensity", "Filter Type", "Affected Pixels (Task2)", "Corrected Pixels (Task4)", "Corrupted Pixels (Task5)"])

    print("\nRunning iterations... (This may take a moment)\n")
    print(f"{'Noise':<12} | {'Int':<5} | {'Filter':<15} | {'Affected':<8} | {'Corrected':<8} | {'Corrupted':<8}")
    print("-" * 75)

    # Loop through all combinations
    for n_type in noise_types:
        for p in intensities:
            # 1. Add Noise (Task 1)
            g = add_noise(f, n_type, intensity=p)
            
            for f_type in filter_types:
                # Special tuning for Contraharmonic Filter:
                # Q = positive removes Pepper noise.
                # Q = negative removes Salt noise.
                Q_val = 1.5 
                if n_type == "salt": 
                    Q_val = -1.5 # Adaptive tuning for Salt noise
                
                # 3. Apply Filter (Task 3)
                k = apply_filter(g, f_type, kernel_size=kernel_size, Q=Q_val, d=2)
                
                # Calculate Metrics (Tasks 2, 4, 5)
                affected, corrected, corrupted = calculate_metrics(f, g, k)
                
                # Print result to console
                print(f"{n_type:<12} | {p:<5} | {f_type:<15} | {affected:<8} | {corrected:<8} | {corrupted:<8}")
                
                # Save result to CSV
                csv_writer.writerow([n_type, p, f_type, affected, corrected, corrupted])
                
                # --- Save Visual Results ---
                # We save images for 0.1 and 0.2 intensities to keep the folder manageable
                if p == 0.1 or p == 0.2: 
                    filename = f"{n_type}_int{str(p).replace('.','')}_{f_type}.png"
                    save_path = os.path.join(output_dir, filename)
                    
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(131)
                    plt.imshow(f, cmap='gray')
                    plt.title('Original (f)')
                    plt.axis('off')
                    
                    plt.subplot(132)
                    plt.imshow(g, cmap='gray')
                    plt.title(f'Noisy (g)\n{n_type} ({p})')
                    plt.axis('off')
                    
                    plt.subplot(133)
                    plt.imshow(k, cmap='gray')
                    plt.title(f'Filtered (k)\n{f_type}')
                    plt.axis('off')
                    
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close() # Close figure to free memory

    csv_file.close()
    print(f"\n✅ All tasks completed successfully.")
    print(f"📂 Images and 'metrics_table.csv' have been saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    main()