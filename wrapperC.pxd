cdef extern from 'numerical_backend/static.c':
    void vegas(double *regn, int ndim, double (*fxn)(double *, double), int init,
                unsigned long ncall, int itmx, int nprn, double *tgral, double *sd,
                double *chi2a)

    void a_vegas(void *realf, double *regn, int ndim, double (*fxn)(void *, double *, double, int), int init,
                unsigned long ncall, int itmx, int nprn, double *tgral, double *sd,
                double *chi2a)

 
