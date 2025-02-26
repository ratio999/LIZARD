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
noisy_signal = original_signal + np.random.normal(0, 0.51, len(t))



# Wavelet transform parameters
wavelet = 'db4'
max_level = pywt.dwt_max_level(len(noisy_signal), pywt.Wavelet(wavelet).dec_len)
level = min(5, max_level)
print(level)
def threshold(coeffs, threshold_value, mode='soft'):
    thresholded_coeffs = []
    for i, coeff in enumerate(coeffs):
        if i == 0:  # Skip the approximation coefficients (low-pass component)
            thresholded_coeffs.append(coeff)
            continue
        if mode == 'soft':
            thresholded_coeffs.append(pywt.threshold(coeff, threshold_value, mode='soft'))
        elif mode == 'hard':
            thresholded_coeffs.append(pywt.threshold(coeff, threshold_value, mode='hard'))
    return thresholded_coeffs

coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)
threshold_value = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(noisy_signal)))  # Universal threshold
thresholded_coeffs = threshold(coeffs, threshold_value, mode='soft')
# Compute wavelet coefficients

#print(coeffs.shape)
def polya_tree_threshold(coeffs, alpha=1.0, beta=1.0, c=2.0):
    """
    Implement Polya tree based thresholding
    alpha, beta: parameters for Beta distribution
    c: concentration parameter
    """
    selected_coeffs = [coeffs[0]]
    selected_coeffs1 = [coeffs[0]]
 
    # Handle approximation coefficients
    
    # Process detail coefficients using Polya tree
    for level, coef in enumerate(coeffs[1:], 1):
        # Adjust parameters based on level
        level_alpha = alpha * (c**level)
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
        threshold_std = np.sum(coef_energy)/coef_energy.shape[0]
        mask1 = coef_energy>threshold_std
        selected_coeffs.append(coef_selected)
        

        coef_selected1 = coef.copy()
        
        coef_selected1[~mask1] = 0
        selected_coeffs1.append(coef_selected1) 
    return selected_coeffs, selected_coeffs1

# Grid search over concentration parameter
c_values = np.linspace(1.1, 5.0, 20)
errors_polya = []
reconstructions_polya = []
errors_polya1 = []
reconstructions_polya1=[]

for c in c_values:
    selected, selected1 = polya_tree_threshold(coeffs, c=c)
    rec_signal = pywt.waverec(selected, wavelet)
    rec_signal1 = pywt.waverec(selected1, wavelet)
    
    # Ensure reconstruction length matches
    if len(rec_signal) > len(t):
        rec_signal = rec_signal[:len(t)]
    elif len(rec_signal) < len(t):
        rec_signal = np.pad(rec_signal, (0, len(t) - len(rec_signal)), mode='edge')
    if len(rec_signal1) > len(t):
        rec_signal1 = rec_signal1[:len(t)]
    elif len(rec_signal) < len(t):
        rec_signal1 = np.pad(rec_signal1 (0, len(t) - len(rec_signal1)), mode='edge')
    
   
    error = np.mean((original_signal - rec_signal) ** 2)
    error1 = np.mean((original_signal - rec_signal1) ** 2)

    errors_polya.append(error)
    errors_polya1.append(error1)
    reconstructions_polya.append(rec_signal)
    reconstructions_polya1.append(rec_signal1)

opt_index_polya = np.argmin(errors_polya)
opt_c = c_values[opt_index_polya]
opt_rec_signal_polya = reconstructions_polya[opt_index_polya]
opt_index_polya1 = np.argmin(errors_polya1)
opt_c1 = c_values[opt_index_polya1]
#opt_rec_signal_polya1 = reconstructions_polya1[opt_index_polya1]
reconstructed_signal = pywt.waverec(thresholded_coeffs, 'db4')
opt_rec_signal_polya1 = reconstructed_signal
print('Optimal concentration parameter:', opt_c)
print('Minimum MSE (Polya):', errors_polya[opt_index_polya])
print('Minimum MSE (std):', np.mean((original_signal - reconstructed_signal)**2)) 

# Plot grid search results
plt.figure(figsize=(10, 4))
plt.plot(c_values, errors_polya, marker='o')
plt.plot(c_values, errors_polya1, marker='o')

plt.title('Polya Tree Grid Search: MSE vs Concentration Parameter')
plt.xlabel('Concentration Parameter (c)')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

# Plot signal reconstruction and residuals
residuals_polya = original_signal - opt_rec_signal_polya
residuals_polya1 = original_signal - opt_rec_signal_polya1
residuals_models = opt_rec_signal_polya - opt_rec_signal_polya1
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
#plt.plot(t, original_signal, label='Original Signal', alpha=0.8)
#plt.plot(t, noisy_signal, label='Noisy Signal', alpha=0.4)
plt.plot(t, opt_rec_signal_polya, label='Polya Tree Reconstructed', alpha=0.8)
plt.plot(t, opt_rec_signal_polya1, label='Standard Reconstructed', alpha=0.8)
plt.plot(t, noisy_signal, label='Noisy', alpha=0.4)
plt.title('Original vs Polya Tree Reconstructed Signal')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, residuals_polya, 'r', label='Residuals (Original - Polya Reconstructed)', alpha=0.7)
plt.plot(t, residuals_polya1, 'b', label='Residuals (Original - Standard Reconstructed)', alpha=0.7)

plt.title('Residuals (Original - Reconstructed)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.hist(residuals_polya, bins=50, density=True, alpha=0.7, label= 'Polya Tree Reconstruction')
plt.hist(residuals_polya1, bins=50, density=True, alpha=0.7, label='Standard Reconstruction')

plt.title('Histogram of Residuals (Original - Reconstructed)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Compare coefficient retention between methods
def count_nonzero_coeffs(coeffs_list):
    return sum(np.count_nonzero(c) for c in coeffs_list)

# Get coefficients from both methods
polya_coeffs = polya_tree_threshold(coeffs, c=opt_c)[0]
#print(polya_coeffs.shape)
print('\
Coefficient retention comparison:')
print('Polya Tree nonzero coefficients:', count_nonzero_coeffs(polya_coeffs))
print('Soft thresholding nonzero coefficients:', count_nonzero_coeffs(thresholded_coeffs))

print('\
Residuals (Original - Polya Reconstructed) statistics:')
print('Mean:', np.mean(residuals_polya))
print('Std:', np.std(residuals_polya))
print('Min:', np.min(residuals_polya))
print('Max:', np.max(residuals_polya))
print('\
Residuals (Original - Soft thresholding-Reconstructed) statistics:')
print('Mean:', np.mean(residuals_polya1))
print('Std:', np.std(residuals_polya1))
print('Min:', np.min(residuals_polya1))
print('Max:', np.max(residuals_polya1))


print('Done.')
