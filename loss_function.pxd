# loss_function.pxd
# Declarations exported to other Cython modules (e.g. your SR engine).
#
# Usage in another .pyx:
#   from losses.loss_function cimport LossFunction, CustomLoss

cimport numpy as cnp

cdef class LossFunction:
    cdef  str  name
    cdef  bint has_grad

    cpdef double evaluate_scalar(self, double y_true, double y_pred)
    cpdef double grad_scalar   (self, double y_true, double y_pred)
    cpdef double evaluate_array(self, cnp.ndarray[cnp.float64_t, ndim=1] y_true, cnp.ndarray[cnp.float64_t, ndim=1] y_pred)
    cpdef cnp.ndarray[cnp.float64_t, ndim=1] grad_array(self, cnp.ndarray[cnp.float64_t, ndim=1] y_true, cnp.ndarray[cnp.float64_t, ndim=1] y_pred)


cdef class CustomLoss(LossFunction):
    cdef object _fn
    cdef object _grad_fn
    cdef bint   _vectorized

    cpdef double evaluate_scalar(self, double y_true, double y_pred)
    cpdef double grad_scalar   (self, double y_true, double y_pred)
    cpdef double evaluate_array(self, cnp.ndarray[cnp.float64_t, ndim=1] y_true, cnp.ndarray[cnp.float64_t, ndim=1] y_pred)
    cpdef cnp.ndarray[cnp.float64_t, ndim=1] grad_array(self, cnp.ndarray[cnp.float64_t, ndim=1] y_true, cnp.ndarray[cnp.float64_t, ndim=1] y_pred)


cdef class MAELoss(LossFunction):
    cpdef double evaluate_scalar(self, double y_true, double y_pred)
    cpdef double grad_scalar   (self, double y_true, double y_pred)


cdef class HuberLoss(LossFunction):
    cdef double delta
    cpdef double evaluate_scalar(self, double y_true, double y_pred)
    cpdef double grad_scalar   (self, double y_true, double y_pred)


cdef cnp.ndarray[cnp.float64_t, ndim=1] _as_f64(arr)
