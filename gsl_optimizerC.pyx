# gsl_optimizer.pyx
cimport cython
from libc.math cimport sqrt


# Funzione obiettivo per la minimizzazione (quadratica)
def objective_function(double *params):
    return (params[0] - 2)**2 + (params[1] - 3)**2  # Minimizziamo (x - 2)^2 + (y - 3)^2

# Funzione per eseguire la minimizzazione con Nelder-Mead
def nelder_mead_minimize():
    cdef int dim = 2  # Dimensione del problema (2D)
    cdef gsl_multimin_function obj_func
    cdef gsl_multimin_fminimizer *minimizer
    cdef double *params = <double *> malloc(dim * sizeof(double))
    cdef double *step = <double *> malloc(dim * sizeof(double))
    cdef int status
    
    params[0] = 0.0  # Impostiamo il valore iniziale
    params[1] = 0.0
    step[0] = 0.1  # Passo iniziale
    step[1] = 0.1

    # Impostiamo la funzione obiettivo
    obj_func.f = objective_function
    obj_func.n = dim
    obj_func.params = NULL

    # Creiamo l'oggetto minimizzatore
    minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex, dim)

    # Minimizziamo
    status = gsl_multimin_fminimizer_minimum(minimizer)

    if status == 0:
        print("Minimizzazione Nelder-Mead riuscita!")
        print("Valore ottimale:", params[0], params[1])
    else:
        print("Errore nella minimizzazione")
    
    gsl_multimin_fminimizer_free(minimizer)

# Funzione per eseguire la minimizzazione con Levenberg-Marquardt
def levenberg_marquardt_minimize():
    cdef int dim = 2  # Dimensione del problema (2D)
    cdef gsl_multimin_function obj_func
    cdef gsl_multimin_fminimizer *minimizer
    cdef double *params = <double *> malloc(dim * sizeof(double))
    cdef double *step = <double *> malloc(dim * sizeof(double))
    cdef int status
    
    params[0] = 0.0  # Impostiamo il valore iniziale
    params[1] = 0.0
    step[0] = 0.1  # Passo iniziale
    step[1] = 0.1

    # Impostiamo la funzione obiettivo
    obj_func.f = objective_function
    obj_func.n = dim
    obj_func.params = NULL

    # Creiamo l'oggetto minimizzatore
    minimizer = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex, dim)

    # Minimizziamo
    status = gsl_multimin_fminimizer_minimum(minimizer)

    if status == 0:
        print("Minimizzazione Levenberg-Marquardt riuscita!")
        print("Valore ottimale:", params[0], params[1])
    else:
        print("Errore nella minimizzazione")
    
    gsl_multimin_fminimizer_free(minimizer)

