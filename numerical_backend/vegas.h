#ifndef VEGAS_H
#define VEGAS_H

void vegas(double *regn, int ndim, double (*fxn)(double *, double), int init, 
		unsigned long ncall, int imx, int nprn, double *tgral, double *sd,
		double *chi2a);

#endif
