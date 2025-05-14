import numpy as np
import matplotlib.pyplot as plt
import pywt

def generate_signal():
    t = np.linspace(0, 10, 1000, endpoint=False)
    signal = (11-t)**(-0.75)*np.sin(20*(11-t)**(5/8))  # 50 Hz + chirp
    return t, signal

# Perform Continuous Wavelet Transform (CWT)
def compute_cwt(signal, scales, wavelet='cmor'):
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1/1000)
    return coefficients, frequencies
scales = np.arange(1, 128)     
t_gw_corrected, signal_gw_corrected = generate_signal()

# Perform CWT on the corrected GW signal
coefficients_gw_corrected, frequencies_gw_corrected = compute_cwt(signal_gw_corrected, scales)

# Plot the corrected time-frequency representation
plt.figure(figsize=(10, 6))
plt.contourf(t_gw_corrected, frequencies_gw_corrected, np.abs(coefficients_gw_corrected), extend='both', cmap='viridis')
plt.title('Time-Frequency Representation (CWT) for Corrected GW Inspiral Signal', fontsize=20, pad=15)
plt.xlabel('Time (s)', fontsize=16, labelpad=10)
plt.ylabel('Frequency (Hz)', fontsize=16, labelpad=10)
plt.yscale('log')
plt.colorbar(label='Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()
print('Displayed corrected time-frequency representation for GW signal.')
def plot_frequency_vs_time(t, coefficients, frequencies):
    # Find the dominant frequency at each time step
    dominant_frequencies = frequencies[np.argmax(np.abs(coefficients), axis=0)]

    # Plot the dominant frequency as a function of time
    plt.figure(figsize=(10, 4))
    plt.plot(t, dominant_frequencies, color='blue')
    plt.title("Dominant Frequency as a Function of Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call the function to plot frequency vs time
plot_frequency_vs_time(t_gw_corrected, coefficients_gw_corrected, frequencies_gw_corrected)
signal = signal_gw_corrected
t = t_gw_corrected
renorm = max(np.correlate(signal, signal, mode='full'))**0.5
signal /= renorm

from scipy.signal import hilbert, savgol_filter
from scipy.interpolate import CubicSpline
# Define the spline regions
spline_length = 1000  # Number of samples to spline at the beginning and end
fs = 16384
# Create a spline function for the beginning
x_begin = np.linspace(0, 1, spline_length)
spline_begin = CubicSpline([0, 1], [0, 1], bc_type=((1, 0), (2, 0)))  # Smooth transition from 0 to 1

# Create a spline function for the end
x_end = np.linspace(0, 1, spline_length)
spline_end = CubicSpline([0, 1], [1, 0], bc_type=((1, 0), (2, 0)))  # Smooth transition from 1 to 0

# Apply the spline to the signal
splined_signal = signal.copy()
splined_signal[:spline_length] *= spline_begin(x_begin)  # Spline the beginning
#splined_signal[len(signal)-spline_length:] *= spline_end(x_end)  # Spline the end

analytic_signal = hilbert(splined_signal)

# Compute the instantaneous phase
instantaneous_phase = np.unwrap(np.angle(analytic_signal))

# Compute the instantaneous frequency
instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

plt.plot(t[:-1], instantaneous_frequency)
#plt.plot(t, frequencies_gw_corrected)
dominant_frequencies = frequencies_gw_corrected[np.argmax(np.abs(coefficients_gw_corrected), axis=0)]
#plt.plot(t[:-1], dominant_frequencies[:-1])
plt.show()

