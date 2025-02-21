# Improved approach: Depth-adaptive thresholding with optimization for reconstruction

import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.stats import beta as bdist

N=1000
t = np.arange(0, 10, 0.01)
original_signal = (11-t)**(-0.75)*np.sin(20*(11-t)**(5/8))
#t = t[::-N//10]
#original_signal = original_signal[::-N//10]
noisy_signal = original_signal + 0.1*np.random.normal(0, 1, len(t))



# Wavelet transform parameters
wavelet = 'db4'
max_level = pywt.dwt_max_level(len(noisy_signal), pywt.Wavelet(wavelet).dec_len)
level = min(5, max_level)

# Compute wavelet coefficients
coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)
#print(coeffs.shape)
def polya_tree_threshold(coeffs, alpha=1.0, beta=1.0, c=2.0):
    """
    Implement Polya tree based thresholding
    alpha, beta: parameters for Beta distribution
    c: concentration parameter
    """
    selected_coeffs = [coeffs[0]]
    
    # Handle approximation coefficients
    
    
    # Process detail coefficients using Polya tree
    for level, coef in enumerate(coeffs[1:], 1):
        # Adjust parameters based on level
        level_alpha = alpha * ((c) ** level)
        level_beta = beta * (c ** level)
        
        # Normalize coefficients
        coef_energy = np.abs(coef)
        if np.sum(coef_energy) > 0:
            coef_prob = coef_energy / np.sum(coef_energy)
        else:
            coef_prob = np.zeros_like(coef_energy)
            
        # Calculate posterior parameters
        post_alpha = level_alpha + coef_prob * len(coef)
        post_beta = level_beta + (1 - coef_prob) * len(coef)
        
        # Calculate threshold using posterior mean
        threshold = post_alpha / (post_alpha + post_beta)
        
        # Apply threshold
        mask = coef_prob > threshold.mean()  # Using mean threshold for simplicity
        coef_selected = coef.copy()
        coef_selected[~mask] = 0
        
        selected_coeffs.append(coef_selected)
    
    return selected_coeffs

# Grid search over concentration parameter
c_values = np.linspace(1.1, 5.0, 20)
errors_polya = []
reconstructions_polya = []

for c in c_values:
    selected = polya_tree_threshold(coeffs, c=c)
    rec_signal = pywt.waverec(selected, wavelet)
    
    # Ensure reconstruction length matches
    if len(rec_signal) > len(t):
        rec_signal = rec_signal[:len(t)]
    elif len(rec_signal) < len(t):
        rec_signal = np.pad(rec_signal, (0, len(t) - len(rec_signal)), mode='edge')
    
    error = np.mean((original_signal - rec_signal) ** 2)
    errors_polya.append(error)
    reconstructions_polya.append(rec_signal)

opt_index_polya = np.argmin(errors_polya)
opt_c = c_values[opt_index_polya]
opt_rec_signal_polya = reconstructions_polya[opt_index_polya]

print('Optimal concentration parameter:', opt_c)
print('Minimum MSE (Polya):', errors_polya[opt_index_polya])

# Plot grid search results
plt.figure(figsize=(10, 4))
plt.plot(c_values, errors_polya, marker='o')
plt.title('Polya Tree Grid Search: MSE vs Concentration Parameter')
plt.xlabel('Concentration Parameter (c)')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

# Plot signal reconstruction and residuals
residuals_polya = original_signal - opt_rec_signal_polya

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, original_signal, label='Original Signal', alpha=0.8)
plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.4)
plt.plot(t, opt_rec_signal_polya, label='Polya Tree Reconstructed', alpha=0.8)
plt.title('Original vs Polya Tree Reconstructed Signal')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, residuals_polya, 'r', label='Residuals (Original - Polya Reconstructed)', alpha=0.7)
plt.title('Residuals (Original - Polya Reconstructed)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.hist(residuals_polya, bins=50, density=True, alpha=0.7)
plt.title('Histogram of Residuals (Original - Polya Reconstructed)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare coefficient retention between methods
def count_nonzero_coeffs(coeffs_list):
    return sum(np.count_nonzero(c) for c in coeffs_list)

# Get coefficients from both methods
polya_coeffs = polya_tree_threshold(coeffs, c=opt_c)
adaptive_coeffs = adaptive_polya_selection(coeffs, opt_fac_refined)

print('\
Coefficient retention comparison:')
print('Polya Tree nonzero coefficients:', count_nonzero_coeffs(polya_coeffs))
print('Adaptive nonzero coefficients:', count_nonzero_coeffs(adaptive_coeffs))

print('\
Residuals (Original - Polya Reconstructed) statistics:')
print('Mean:', np.mean(residuals_polya))
print('Std:', np.std(residuals_polya))
print('Min:', np.min(residuals_polya))
print('Max:', np.max(residuals_polya))

print('Done.')