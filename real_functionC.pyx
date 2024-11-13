import numpy as np
cimport numpy as cnp
import sympy as sp
cimport sympy as csp
from libc.stdlib cimport malloc, free  # Importa malloc e free per la gestione della memoria se necessario

cdef class RealFunctionC:

    def __init__(self, object func, object expr, cnp.ndarray[cnp.float32_t, ndim=1] domain, list params=None):
        """Inizializza la classe RealFunction."""
        self.func = func
        self.expr = expr  # Non serve sp.sympify qui, poiché expr è già un oggetto Sympy
        self.description = str(expr)  # Imposta la descrizione come stringa dell'espressione
        self.regular = False  # Inizializza a False
        self.params = params
        self.domain = domain
        '''
        if isinstance(domain, (np.ndarray)):  # Controlla se domain è un array
            if self.is_defined(domain).any():
                self.description = str(expr)
        elif domain is None:
            self.description = str(expr)
        '''
        if not isinstance(domain, (np.ndarray)):  # Se domain è scalare
            if self.is_defined(domain):
                self.description = str(expr)
                self.regular = True
            else:
                self.description = None
               
        else:
            self.description = str(expr)
            self.regular = True
    
    cpdef get_domain(self):
        return self.domain

    def evaluate(self,   x):
        """Valuta l'espressione simbolica per un array di input x, sostituendo x con i valori dell'array."""
 
        cdef int n = x.size

        cdef cnp.ndarray[cnp.float32_t, ndim=1] results = np.zeros(n, dtype=np.float32)
        cdef object substituted_expr

        # Definisci il simbolo per x
        cdef object x_sym = sp.symbols('x')
      
        # Valuta l'espressione per ogni valore in x
        for i in range(n):
            substituted_expr = self.expr.subs(x_sym, x[i])
            #print('sub= ', substituted_expr)
            if substituted_expr.is_real and substituted_expr.is_finite:
                results[i] = float(substituted_expr)  # Converti l'espressione sostituita in float e memorizza
            else:
                results[i] = np.nan  # Converti l'espressione sostituita in float e memorizza

        return results

    cpdef is_param(self):
        cdef cnp.ndarray[cnp.float32_t, ndim=1] y
        
    
        y = self.evaluate(self.domain)
        return np.all(np.isclose(y, y[0]))

    cpdef mutate_parameters(self):
        """Modifica i valori dei parametri in modo casuale."""
        for i in range(len(self.params)):
            # Esempio di mutazione: incremento casuale
            self.params[i] += np.random.normal(0, 0.1)  # Aggiunge un rumore normale

    def is_defined(self, cnp.ndarray[cnp.float32_t, ndim=1] x):
        """Verifica se la funzione è definita per un array di input x."""
        cdef int n = x.size
        cdef list results = []
        #cdef object results = np.zeros(n, dtype=np.bool_)
        cdef object x_sym = sp.symbols('x')
        def check_single_value(cnp.float64_t x_val):
            try:
                expr_eval = self.expr.subs(x_sym, x_val)
                return expr_eval.is_real and expr_eval.is_finite
            except ValueError as e:
                return False

        if isinstance(x, np.ndarray):
            for i in range(n):
                results.append(check_single_value(x[i]))  # Aggiungi il risultato alla lista

            return np.array(results, dtype=np.bool_)
        else:
            return check_single_value(x)
        '''
        for i in range(n):
            try:
                # Verifica se la funzione può essere valutata
                _ = self.func(x[i])
                results[i] = True
            except Exception:
                results[i] = False

        return results
        '''
    cpdef derivative(self,  x,  h=1e-5):
        """Calcola la derivata della funzione in x usando il metodo delle differenze finite."""
        return (self.evaluate(x + h) - self.evaluate(x - h)) / (2 * h)

    def __call__(self, cnp.ndarray[cnp.float32_t, ndim=1] x):
        """Permette di chiamare l'oggetto come una funzione."""
        return self.evaluate(x)

    @classmethod
    def from_string(cls, expr_str,  cnp.ndarray[cnp.float32_t, ndim=1] x):
        """Crea un'istanza di RealFunction da una stringa che rappresenta un'espressione simbolica."""
        #print(f"expr_str type: {type(expr_str)}")  # Dovrebbe stampare <class 'str'>
        #print(f"x type: {type(x)}")  # Dovrebbe stampare <class 'numpy.ndarray'>
        
        cdef object x_sym = sp.symbols('x')  # Definisci un simbolo di Sympy per x
        cdef object expr = sp.sympify(expr_str)  # Converte la stringa in un'espressione simbolica

        # Crea una funzione lambda per valutare l'espressione simbolica
        def func(cnp.float32_t input_x):
            return np.array(sp.lambdify(x_sym, expr)(input_x), dtype=np.float32)

        return cls(func, expr, x)  # Passa x come domain

    def get_description(self):
        """Restituisce la descrizione della funzione."""
        return self.description

    def __add__(self, RealFunctionC other):
        """Addition of two RealFunctions."""
        cdef object new_expr = self.expr + other.expr

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            return self.evaluate(x) + other.evaluate(x)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __sub__(self, RealFunctionC other):
        """Subtraction of two RealFunctions."""
        cdef object new_expr = self.expr - other.expr

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            return self.evaluate(x) - other.evaluate(x)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __mul__(self, RealFunctionC other):
        """Multiplication of two RealFunctions."""
        cdef object new_expr = self.expr * other.expr

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            return self.evaluate(x) * other.evaluate(x)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __truediv__(self, RealFunctionC other):
        """Division of two RealFunctions with protection against division by zero."""
        cdef object new_expr = self.expr / other.expr

        def safe_div(cnp.ndarray[cnp.float32_t, ndim=1] x):
            cdef object numerator = self.evaluate(x)
            cdef object denominator = other.evaluate(x)
            return np.where(denominator == 0, np.inf, numerator / denominator)

        return RealFunctionC(safe_div, new_expr, self.domain)

    def compose(self, RealFunctionC other):
        """Composition of two RealFunctions (f1(f2(x)))."""
        cdef object x_sym = sp.symbols('x')
        cdef object new_expr = self.expr.subs(x_sym, other.expr)

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            cdef object is_defined_mask = other.is_defined(x)
            cdef object evaluated_inner = other.evaluate(x)
            cdef object result = np.zeros_like(x)
            result[:] = np.nan
            result[is_defined_mask] = self.evaluate(evaluated_inner[is_defined_mask])
            return result

        return RealFunctionC(new_func, new_expr, self.domain)

    def __pow__(self, RealFunctionC other):
        """Exponentiation of a RealFunction."""
        cdef object new_expr = self.expr ** other.expr

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            return self.evaluate(x) ** other.evaluate(x)

        return RealFunctionC(new_func, new_expr, self.domain)

    cpdef fitness(self, x, y_true):
        predictions = self.evaluate(x)
        mse = np.mean((predictions - y_true) ** 2)
        complexity = len(self.params)  # O qualsiasi misura di complessità
        return mse + complexity * 0.01  # Penalizzazione per complessità
