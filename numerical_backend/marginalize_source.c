#include <math.h>
#include <stdlib.h>
#include<stdio.h>
#include "./vegas.h"

static double marginalize_posterior(double *posterior_values, int n, int num_samples) {

	double sum = 0.0;
	for (int i = 0; i < num_samples; i++){
		sum += posterior_values[i];
		// printf("%d -> %f \n", i, sum);
	}
	return sum/num_samples;
}


static void static_vegas(double regn[], int ndim, double (*fxn)(double [], double), int init,
		unsigned long ncall, int itmx, int nprn, double *tgral, double *sd,
		double *chi2a){

	vegas(regn[], ndim,  fxn, init,  ncall,  itmx,  nprn, tgral, sd, chi2a);

	}

