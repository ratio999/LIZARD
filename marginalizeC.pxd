
cdef double marginalize_posterior(double *posterior_values, int n, int num_samples)
cdef int vegas(double *regn, int ndim, double (*fxn)(double*, double),
               int init, unsigned long ncall, int itmx, int nprn,
               double *tgral, double *sd, double *chi2a)

 
