import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# Signal setup
N=1000
t = np.arange(0, 10, 0.001)
original_signal = (11-t)**(-0.75)*np.sin(20*(11-t)**(5/8))
#t = t[::-N//10]
#original_signal = original_signal[::-N//10]
noisy_signal = original_signal + 18*np.random.normal(0, 0.1, len(t))

# Wavelet transform parameters
wavelet = 'db4'
max_level = pywt.dwt_max_level(len(noisy_signal), pywt.Wavelet(wavelet).dec_len)
level = min(5, max_level)

# Compute wavelet coefficients
coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)

def polya_tree_threshold(coeffs, alpha=1.0, beta=1.0, c=2.0, depth=0, max_depth=3):
    """
    Implement Polya tree based thresholding with explicit partitioning.
    alpha, beta: parameters for Beta distribution.
    c: concentration parameter for depth-dependent shrinkage.
    depth: current depth of the tree.
    max_depth: maximum depth of the tree.
    """
    if depth == max_depth:
        # Base case: apply thresholding to the current partition
        coef_energy = np.abs(coeffs)
        if np.sum(coef_energy) > 0:
            coef_prob = coef_energy / np.sum(coef_energy)
        else:
            coef_prob = np.zeros_like(coef_energy)
        
        # Compute posterior parameters
        post_alpha = alpha * (c ** depth) + coef_prob * len(coeffs)
        post_beta = beta * (c ** depth) + (1 - coef_prob) * len(coeffs)
        
        # Compute posterior mean
        posterior_mean = post_alpha / (post_alpha + post_beta)
        
        # Apply threshold
        threshold = posterior_mean.mean()  # Use mean threshold for simplicity
        mask = coef_prob > threshold
        coef_selected = coeffs.copy()
        coef_selected[~mask] = 0
        
        return coef_selected
    else:
        # Recursive case: split the coefficients and process each partition
        mid = len(coeffs) // 2
        left = polya_tree_threshold(coeffs[:mid], alpha, beta, c, depth + 1, max_depth)
        right = polya_tree_threshold(coeffs[mid:], alpha, beta, c, depth + 1, max_depth)
        return np.concatenate([left, right])

# Grid search over concentration parameter
c_values = np.linspace(1.1, 5.0, 20)
errors_polya = []
reconstructions_polya = []

for c in c_values:
    selected = []
    selected.append(coeffs[0])  # Keep approximation coefficients
    for coef in coeffs[1:]:
        selected.append(polya_tree_threshold(coef, c=c))
    
    rec_signal = pywt.waverec(selected, wavelet)
    
    # Ensure reconstruction length matches
    if len(rec_signal) > len(t):
        rec_signal = rec_signal[:len(t)]
    elif len(rec_signal) < len(t):
        rec_signal = np.pad(rec_signal, (0, len(t) - len(rec_signal)), mode='edge')
    
    error = np.mean((original_signal - rec_signal) ** 2)
    errors_polya.append(error)
    reconstructions_polya.append(rec_signal)

# Find optimal concentration parameter
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
plt.plot(t, original_signal, label='Original Signal', alpha=0.7)
plt.plot(t, opt_rec_signal_polya, label='Polya Tree Reconstructed', alpha=0.7)
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

# Coefficient retention comparison
def count_nonzero_coeffs(coeffs_list):
    return sum(np.count_nonzero(c) for c in coeffs_list)

polya_coeffs = [coeffs[0]] + [polya_tree_threshold(coef, c=opt_c) for coef in coeffs[1:]]
print('Polya Tree nonzero coefficients:', count_nonzero_coeffs(polya_coeffs))

# Residual statistics
print('\
Residuals (Original - Polya Reconstructed) statistics:')
print('Mean:', np.mean(residuals_polya))
print('Std:', np.std(residuals_polya))
print('Min:', np.min(residuals_polya))
print('Max:', np.max(residuals_polya))

print('Done.')