from wrapperC cimport vegas, a_vegas
import numpy as np
cimport numpy as cnp
import ctypes as ct
from real_functionC cimport RealFunctionC
# cython: language_level=3
from cpython.ref cimport Py_INCREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer

from cpython.ref cimport Py_INCREF, Py_DECREF

cdef void* python_object_to_void_ptr(object py_obj):
    """Converte un oggetto Python in un puntatore void*"""
    Py_INCREF(py_obj)  # Incrementa il reference count per evitare la garbage collection
    return <void*> py_obj  # Cast esplicito a void*

cdef object void_ptr_to_python_object(void* ptr):
    """Riconverte un puntatore void* in un oggetto Python"""
    return <object> ptr  # Cast diretto da void* a object



cdef double my_function(void * cl, double *arr, double val, int ndim) noexcept :
    cdef double result = 1.0
    for i in range(0,ndim):
      result *= np.exp(arr[i]+2) + 1e-6
    return result

def main():
      cdef RealFunctionC rf = RealFunctionC.from_string('x', np.array([0.0]))
      cdef void* ptr = python_object_to_void_ptr(rf)
      cdef double[:] regn = np.array([0.0, 0.0, np.pi, np.pi])
      cdef double[:] tgral = np.array([0.0])
      cdef double[:] sd = np.array([0.0])
      cdef double[:] chi2a = np.array([0.0])
      a_vegas(ptr, &regn[0], 2, my_function,
      0, 100000, 5, 0, &tgral[0], &sd[0], &chi2a[0])
      print(*tgral, *sd, *chi2a)
      #print(tup)
