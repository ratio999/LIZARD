import numpy as np
from scipy.integrate import solve_ivp
from pysr import PySRRegressor
import matplotlib.pyplot as plt

# Define the true dynamics (dx/dt = x^y)
def true_dynamics(t, x, y):
    return x**y
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 2.998e8      # Speed of light (m/s)
M_sun = 1.98847e30  # Solar mass (kg)
chirp_mass = 1.2 * M_sun  # Chirp mass (example value)

# Define the true dynamics (df/dt)
def true_dynamics(t, x, y):
    return  x**(y)
# Generate synthetic data
t_span = (0, 10)  # Time range
t_eval = np.linspace(0, 10, 1000)  # Time points
x0 = [1.0]  # Initial condition
y = 11.0/3.0  # Example non-integer exponent

# Solve the differential equation
sol = solve_ivp(true_dynamics, t_span, x0, t_eval=t_eval, args=(y,))

# Extract data
t = sol.t
x = sol.y[0]
dx_dt = true_dynamics(t, x, y)  # True derivative

# Prepare data for PySR
noise_level = 0.1  # Adjust the noise level (e.g., 10% of the signal)
dx_dt_noisy = dx_dt + noise_level * np.std(dx_dt) * np.random.randn(len(dx_dt))

# Prepare data for PySR
X = x.reshape(-1, 1)  # Input: x
y_target = dx_dt_noisy.ravel() 
#X = x.reshape(-1, 1)  # Input: x
#y_target = dx_dt.ravel()  # Target: dx/dt

# Define the PySR model
model = PySRRegressor(
    niterations=200,  # Number of iterations
    binary_operators=["+", "-", "*", "/", "^"],  # Allowed operators
    unary_operators=["exp", "log", "sqrt", "cos", "sin"],  # Allowed functions
    populations=20,  # Number of populations
    population_size=50,  # Size of each population
    maxsize=20,  # Maximum size of the expression
    parsimony=0.01,  # Encourages simpler models
    progress=True,  # Show progress bar
)

# Fit the model
model.fit(X, y_target)

# Print the best symbolic expression
print("Discovered equation:")
print(model.sympy())

# Predict dx/dt using the discovered equation
dx_dt_pred = model.predict(X)
print(dx_dt_noisy.shape)
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, dx_dt_noisy, color='blue', alpha=0.6, label='Data')
plt.plot(t, dx_dt_pred, color='red', linestyle='--', label=''.join(('Predicted: ', str(model.sympy()))))
plt.xlabel('Time (t)')
plt.ylabel('df/dt')
plt.title('Symbolic Regression: True vs Predicted df/dt')
plt.legend()
plt.show()