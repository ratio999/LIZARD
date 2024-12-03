cimport numpy as cnp
import numpy as np
cimport function_treeC
cimport real_functionC
from fitnessC cimport calculate_likelihood
cdef calculate_posterior(function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] data):
    # Valuta il modello rappresentato dall'albero sui dati
    #cdef cnp.ndarray[cnp.float64_t, ndim=1] predictions = np.array([tree.evaluate(x) for x in X])
        
    cdef cnp.float64_t alpha = 0.01
    cdef cnp.float64_t sigma = 1.00

    cdef cnp.float64_t sse = calculate_likelihood(tree, data) 
    
    
    log_likelihood = -sse / (2 * sigma**2)
    
    
    complexity = tree.node_count() 
    log_prior = -alpha * complexity
    
    # Logaritmo del posterior
    log_posterior = log_likelihood + log_prior
    
    return log_posterior

    