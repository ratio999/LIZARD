#include <stdio.h>
#include <math.h>
#include "./vegas.h"
// Dichiarazione della funzione vegas
//void vegas(float regn[], int ndim, float (*fxn)(float [], float), int init,
//           unsigned long ncall, int itmx, int nprn, float *tgral, float *sd, float *chi2a);

// Funzione di esempio da integrare
float my_function(float x[], float weight) {
    float result = 1.0;  // Cambia questa funzione per test diversi
    //for (int i = 0; i < 20; i++) {
    //   result *= exp(x[i]);
    //}
    return result;
}

int main() {
    float regn[4] = {0.0, 0.0, M_PI, M_PI};  // Range di integrazione per x e y
    float tgral, sd, chi2a;

    // Chiamata alla funzione vegas
    vegas(regn, 2, my_function, 0, 10000, 500, 0, &tgral, &sd, &chi2a);

    // Mostra i risultati
    printf("Risultato dell'integrale: %.7g\n", tgral);
    printf("Deviazione standard: %.7g\n", sd);
    printf("Chi^2/IT: %.7g\n", chi2a);

    return 0;
}

