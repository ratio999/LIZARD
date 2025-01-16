#include <math.h>
#include <stdlib.h>
#include<stdio.h>

static double marginalize_posterior(double *posterior_values, int n, int num_samples) {

	double sum = 0.0;
	for (int i = 0; i < num_samples; i++){
		sum += posterior_values[i];
		// printf("%d -> %f \n", i, sum);
	}
	return sum/num_samples;
}

