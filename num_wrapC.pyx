from wrapperC cimport vegas
import numpy as np
cimport numpy as cnp
import ctypes as ct


cdef double my_function(double *arr, double val) noexcept:
    cdef double result = 1.0
    for i in range(0,2):
      result *= np.exp(arr[i]+2) + 1e-6
    return result

def main():
      cdef double[:] regn = np.array([0.0, 0.0, np.pi, np.pi])
      cdef double[:] tgral = np.array([0.0])
      cdef double[:] sd = np.array([0.0])
      cdef double[:] chi2a = np.array([0.0])
      vegas(&regn[0], 2, my_function,
      0, 100000, 5, 0, &tgral[0], &sd[0], &chi2a[0])
      print(*tgral, *sd, *chi2a)
      #print(tup)
