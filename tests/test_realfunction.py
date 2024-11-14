import numpy as np
import sympy as sp
from real_functionC import RealFunctionC

# Definisci un'espressione simbolica in SymPy con variabili e parametri
x = sp.symbols('x')
a0, a1, a2 = sp.symbols('a0 a1 a2')
expr1 = a0 * x**2 + a1 * x + a2
b0, b1, b2 = sp.symbols('b0 b1 b2')
expr2 = b0*sp.exp(-x/b1 + b2)
expr3 = b0
# Definisce il dominio
domain = np.linspace(-10, 10, 100, dtype=np.float32)
print(str(expr1), str(expr2))
# Crea un'istanza di RealFunctionC
real_func1 = RealFunctionC.from_string(str(expr1), domain)
real_func2 = RealFunctionC.from_string(str(expr2), domain)
real_func3 = RealFunctionC.from_string(str(expr3), domain)
print('Is third function constant? ', real_func3.is_param())
real_func =real_func3 + real_func1.compose(real_func2)
# Ottieni variabili e parametri
params = real_func.get_params()
variables = real_func.get_variables()
expr = real_func.get_description()
# Stampa i risultati per il test
print("Espressione:", expr)
print("Parametri identificati:", params)
print("Variabili identificate:", variables)
