from wrapperC cimport a_vegas
import numpy as np
cimport numpy as cnp
import ctypes as ct
from real_functionC cimport RealFunctionC
from utils.pointer_utils cimport python_object_to_void_ptr, void_ptr_to_python_object, void_ptr_to_rf
import time
# cython: language_level=3
from cpython.ref cimport Py_INCREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer

from cpython.ref cimport Py_INCREF, Py_DECREF

cdef void* apython_object_to_void_ptr(object py_obj):
    '''Convert Python object to void* pointer'''
    Py_INCREF(py_obj)  # Incrementa il reference count per evitare la garbage collection
    return <void*> py_obj  # Cast esplicito a void*

cdef object avoid_ptr_to_python_object(void* ptr):
    '''Convert void* pointer to Python object'''
    return <object> ptr  # Cast diretto da void* a object



cdef double my_function(void * cl, double *arr, double val, int ndim) noexcept :
    cdef double result = 1.0
    cdef RealFunctionC realf = void_ptr_to_rf(cl)
    for i in range(0,ndim):
      result *= np.sin(arr[i]+2) + 1e-6
    return result

def main():
      cdef RealFunctionC rf = RealFunctionC.from_string('x', np.array([0.0]))
      cdef void* ptr = apython_object_to_void_ptr(rf)
      cdef void* ptr1 = python_object_to_void_ptr(rf)
      print(void_ptr_to_rf(ptr1).get_description())
      cdef long idum
      cdef double[:] regn, tgral, sd, chi2a
      for i in range(1,25):
        idum = -i*12345
        regn = np.array([0.0]*i + [np.pi]*i)
        tgral = np.array([0.0])
        sd = np.array([0.0])
        chi2a = np.array([0.0])
        time1 = time.time()
        print(time1)
        a_vegas(ptr, &regn[0], i, my_function,
          0, 100000, 5, 0, &tgral[0], &sd[0], &chi2a[0], idum)
        time2 = time.time()
        print(time2)
        print(*tgral, *sd, *chi2a, 'in ', time2 - time1, 's')
        
      #print(tup)
