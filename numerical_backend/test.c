#include <stdio.h>
#include <math.h>
#include "vegas.h"
#include <omp.h>

// Dichiarazione della funzione vegas
//void vegas(float regn[], int ndim, float (*fxn)(float [], float), int init,
//           unsigned long ncall, int itmx, int nprn, float *tgral, float *sd, float *chi2a);

// Funzione di esempio da integrare
double my_function(double x[], double weight) {
    double result = 1.0;  // Cambia questa funzione per test diversi
    for (int i = 0; i < 2; i++) {
      result *= exp(x[i]+2) + 1e-6;
    }
    return result;
}

int main() {
    printf("start");
    double regn[4] = {0.0, 0.0, M_PI, M_PI};  // Range di integrazione per x e y
    printf("%f, %f, %f ", regn[0], regn[1], regn[2]);
    double tgral, sd, chi2a;

    // Chiamata alla funzione vegas
    
    vegas(regn, 2, my_function, 0, 100000, 5, 0, &tgral, &sd, &chi2a);
    // Mostra i risultati
    printf("Risultato dell'integrale: %.7g\n", tgral);
    printf("Deviazione standard: %.7g\n", sd);
    printf("Chi^2/IT: %.7g\n", chi2a);

    return 0;
}

