import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import os

# =========================================================
# 1. NOISE GENERATION FUNCTIONS (Task 1)
# =========================================================
def add_noise(image, noise_type, intensity=0.05, mean=0, var=0.01):
    """
    Adds specific noise types to an image.
    Args:
        image: Input grayscale image (numpy array).
        noise_type: 'salt_pepper', 'salt', 'pepper', 'gaussian', 'uniform'.
        intensity: Probability for impulse noise or scale for uniform noise.
        mean: Mean for Gaussian noise.
        var: Variance for Gaussian noise.
    Returns:
        Noisy image (uint8).
    """
    row, col = image.shape
    noisy_image = image.copy()
    
    if noise_type == "salt_pepper":
        # Salt and Pepper: randomly set pixels to 0 or 255
        s_vs_p = 0.5 # Ratio between salt and pepper
        amount = intensity
        
        # Add Salt (White)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 255
        
        # Add Pepper (Black)
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
        
    elif noise_type == "salt":
        # Only Salt (White)
        num_salt = np.ceil(intensity * image.size)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 255

    elif noise_type == "pepper":
        # Only Pepper (Black)
        num_pepper = np.ceil(intensity * image.size)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
        
    elif noise_type == "gaussian":
        # Gaussian / Normal Distribution Noise
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        
        # Normalize, add noise, and clip back to 0-255
        noisy_image = noisy_image / 255.0 + gauss
        noisy_image = np.clip(noisy_image, 0, 1) * 255
        noisy_image = noisy_image.astype(np.uint8)

    elif noise_type == "uniform":
        # Uniform Distribution Noise
        noise = np.random.uniform(-intensity*255, intensity*255, (row, col))
        noisy_image = noisy_image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

# =========================================================
# 2. FILTERING FUNCTIONS (Task 3)
# =========================================================
def apply_filter(noisy_img, filter_type, kernel_size=3, Q=1.5, d=2):
    """
    Applies different spatial filters to remove noise.
    Args:
        filter_type: 'median', 'mean', 'alpha_trimmed', 'contraharmonic'.
        Q: Order of the Contraharmonic filter.
        d: Trim parameter for Alpha-trimmed mean.
    """
    if filter_type == "median":
        # Standard Median Filter (Great for Salt & Pepper)
        return cv2.medianBlur(noisy_img, kernel_size)
    
    elif filter_type == "mean": 
        # Arithmetic Mean Filter (Good for Gaussian)
        return cv2.blur(noisy_img, (kernel_size, kernel_size))
    
    elif filter_type == "alpha_trimmed":
        # Alpha-trimmed Mean Filter (Hybrid of Mean and Median)
        def alpha_trimmed_mean_algo(buffer):
            sort_buf = np.sort(buffer)
            # Trim 'd' pixels from start and end
            trim_slice = sort_buf[d:-d] if d > 0 else sort_buf
            return np.mean(trim_slice)
        
        return generic_filter(noisy_img, alpha_trimmed_mean_algo, size=kernel_size)

    elif filter_type == "contraharmonic":
        # Contraharmonic Mean Filter
        # Formula: sum(pixel^(Q+1)) / sum(pixel^Q)
        img_float = noisy_img.astype(np.float32)
        epsilon = 1e-5 # Avoid division by zero
        
        # Calculate numerator and denominator using boxFilter (which sums neighbors)
        numerator = cv2.boxFilter(np.power(img_float + epsilon, Q + 1), -1, (kernel_size, kernel_size), normalize=False)
        denominator = cv2.boxFilter(np.power(img_float + epsilon, Q), -1, (kernel_size, kernel_size), normalize=False)
        
        result = numerator / (denominator + epsilon)
        return np.clip(result, 0, 255).astype(np.uint8)

    return noisy_img

# =========================================================
# 3. METRIC CALCULATION FUNCTIONS (Tasks 2, 4, 5)
# =========================================================

def count_affected_pixels(f, g):
    """
    Task 2: Count pixels that were affected by noise.
    f: Original Image
    g: Noisy Image
    """
    # Returns count where f is not equal to g
    diff = f != g
    return np.sum(diff)

def count_corrected_pixels(f, g, k):
    """
    Task 4: Count noisy pixels that were fully corrected to their original value.
    f: Original Image
    g: Noisy Image
    k: Filtered Image
    """
    # Condition: Pixel was noisy in g (f!=g) AND is now correct in k (k==f)
    noisy_in_g = (f != g)
    corrected_in_k = (k == f)
    return np.sum(noisy_in_g & corrected_in_k)

def count_corrupted_pixels(f, g, k):
    """
    Task 5: Count clean pixels that were corrupted (changed) by the filter.
    f: Original Image
    g: Noisy Image
    k: Filtered Image
    """
    # Condition: Pixel was clean in g (f==g) BUT changed in k (k!=f)
    clean_in_g = (f == g)
    changed_in_k = (k != f)
    return np.sum(clean_in_g & changed_in_k)

# =========================================================
# 4. MAIN EXECUTION LOOP
# =========================================================
def main():
    # --- Configuration ---
    input_filename = 'input_image.jpg'  # REPLACE with your image path
    
    # Check if image exists
    if not os.path.exists(input_filename):
        print(f"Warning: '{input_filename}' not found. Generating a dummy image for demonstration.")
        # Create a dummy grayscale image (200x200)
        f = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(f, (50, 50), (150, 150), 128, -1)
        cv2.circle(f, (100, 100), 30, 255, -1)
    else:
        f = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)

    # Define experimental parameters
    noise_types = ["salt_pepper", "gaussian", "uniform"]
    filter_types = ["median", "mean", "contraharmonic"]
    
    # Different intensities to test (for the required charts)
    intensities = [0.05, 0.1, 0.2] 
    kernel_size = 3

    # Store results for analysis
    results = []

    # Print Table Header
    print(f"\n{'Noise Type':<15} {'Intensity':<10} {'Filter Type':<15} {'Affected(T2)':<15} {'Corrected(T4)':<15} {'Corrupted(T5)':<15}")
    print("=" * 95)

    # Loop through combinations
    for n_type in noise_types:
        for p in intensities:
            # 1. Add Noise (Task 1)
            g = add_noise(f, n_type, intensity=p)
            
            # 2. Count Affected Pixels (Task 2)
            affected = count_affected_pixels(f, g)

            for f_type in filter_types:
                # 3. Apply Filter (Task 3)
                # Note: For Alpha-trimmed, d needs to be < (kernel_size^2)/2
                # Note: For Contraharmonic, Q=1.5 removes pepper, Q=-1.5 removes salt
                k = apply_filter(g, f_type, kernel_size=kernel_size, Q=1.5, d=2)

                # 4. Count Corrected Pixels (Task 4)
                corrected = count_corrected_pixels(f, g, k)

                # 5. Count Corrupted Pixels (Task 5)
                corrupted = count_corrupted_pixels(f, g, k)

                # Print row in table
                print(f"{n_type:<15} {p:<10} {f_type:<15} {affected:<15} {corrected:<15} {corrupted:<15}")
                
                # Save data for later plotting (if needed)
                results.append({
                    "noise": n_type, "intensity": p, "filter": f_type,
                    "affected": affected, "corrected": corrected, "corrupted": corrupted
                })

                # --- Visualization (Optional: Save 1 example per category) ---
                # We save an image only when intensity is 0.1 to avoid creating too many files
                if p == 0.1:
                    plt.figure(figsize=(12, 4))
                    
                    plt.subplot(131)
                    plt.imshow(f, cmap='gray')
                    plt.title('Original (f)')
                    plt.axis('off')
                    
                    plt.subplot(132)
                    plt.imshow(g, cmap='gray')
                    plt.title(f'Noisy (g)\n{n_type}')
                    plt.axis('off')
                    
                    plt.subplot(133)
                    plt.imshow(k, cmap='gray')
                    plt.title(f'Filtered (k)\n{f_type}')
                    plt.axis('off')
                    
                    # Save the figure
                    save_name = f"Result_{n_type}_{f_type}.png"
                    plt.savefig(save_name)
                    plt.close() 

    print("\nProcessing complete. Check the directory for output images.")

if __name__ == "__main__":
    main()