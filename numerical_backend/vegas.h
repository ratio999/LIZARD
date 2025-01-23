#ifndef VEGAS_H
#define VEGAS_H

void vegas(float regn[], int ndim, float (*fxn)(float [], float), int init, 
		unsigned long ncall, int imx, int nprn, float *tgral, float *sd,
		float *chi2a);

#endif
