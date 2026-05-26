# loss_function.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


import numpy as np
cimport numpy as cnp
from libc.math cimport HUGE_VAL


cdef class LossFunction:

    def __init__(self,  name ,has_grad = False):
        self.name     = name
        self.has_grad = has_grad

    # -- scalar interface (override in subclass) ----------------------------

    cpdef double evaluate_scalar(self, double y_true, double y_pred):
        """Return scalar loss for a single (y_true, y_pred) pair."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement evaluate_scalar()"
        )

    cpdef double grad_scalar(self, double y_true, double y_pred):
        """Return d(loss)/d(y_pred) for a single pair (optional)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide a gradient. "
            "Set has_grad=True and implement grad_scalar() to enable it."
        )

    cpdef double evaluate_array(
        self,
        cnp.ndarray[cnp.float64_t, ndim=1] y_true,
        cnp.ndarray[cnp.float64_t, ndim=1] y_pred,
    ):
        cdef:
            Py_ssize_t i, n = y_true.shape[0]
            double acc = 0.0

        if y_pred.shape[0] != n:
            raise ValueError("y_true and y_pred must have the same length.")

        for i in range(n):
            acc += self.evaluate_scalar(y_true[i], y_pred[i])

        return acc / n

    cpdef cnp.ndarray[cnp.float64_t, ndim=1] grad_array(
        self,
        cnp.ndarray[cnp.float64_t, ndim=1] y_true,
        cnp.ndarray[cnp.float64_t, ndim=1] y_pred,
    ):
        cdef:
            Py_ssize_t i, n = y_true.shape[0]
            cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)

        if y_pred.shape[0] != n:
            raise ValueError("y_true and y_pred must have the same length.")

        for i in range(n):
            out[i] = self.grad_scalar(y_true[i], y_pred[i])

        return out

    # -- unified dispatch (scalar or array) --------------------------------

    def __call__(self, y_true, y_pred, *, gradient=False):
        scalar_input = np.isscalar(y_true) and np.isscalar(y_pred)

        if gradient:
            if not self.has_grad:
                raise NotImplementedError(
                    f"Loss '{self.name}' has no gradient implementation."
                )
            if scalar_input:
                return self.grad_scalar(float(y_true), float(y_pred))
            return np.asarray(
                self.grad_array(
                    _as_f64(y_true),
                    _as_f64(y_pred),
                )
            )

        if scalar_input:
            return self.evaluate_scalar(float(y_true), float(y_pred))
        return self.evaluate_array(
            _as_f64(y_true),
            _as_f64(y_pred),
        )

    def __repr__(self):
        grad_flag = ", has_grad=True" if self.has_grad else ""
        return f"{self.__class__.__name__}(name='{self.name}'{grad_flag})"


# ---------------------------------------------------------------------------
# CustomLoss: wrap any Python callable
# ---------------------------------------------------------------------------

cdef class CustomLoss(LossFunction):

    def __init__(self, fn, grad_fn=None, str name="", bint vectorized=True):
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn)}")
        if grad_fn is not None and not callable(grad_fn):
            raise TypeError(f"grad_fn must be callable or None, got {type(grad_fn)}")

        resolved_name = name or getattr(fn, "__name__", "custom")
        super().__init__(name=resolved_name, has_grad=(grad_fn is not None))

        self._fn         = fn
        self._grad_fn    = grad_fn
        self._vectorized = vectorized

    # -- scalar path -------------------------------------------------------

    cpdef double evaluate_scalar(self, double y_true, double y_pred):
        return <double>self._fn(y_true, y_pred)

    cpdef double grad_scalar(self, double y_true, double y_pred):
        if self._grad_fn is None:
            raise NotImplementedError(
                f"Loss '{self.name}' was created without a grad_fn."
            )
        return <double>self._grad_fn(y_true, y_pred)

    # -- array path --------------------------------------------------------

    cpdef double evaluate_array(
        self,
        cnp.ndarray[cnp.float64_t, ndim=1] y_true,
	cnp.ndarray[cnp.float64_t, ndim=1] y_pred,
    ):
        if self._vectorized:
            # Hand off to NumPy-capable fn; expect a scalar (mean) back
            return <double>self._fn(y_true, y_pred)
        # Element-wise Cython loop (avoids Python overhead per iteration
        # for the dispatcher, but fn itself is still a Python call)
        return LossFunction.evaluate_array(self, y_true, y_pred)

    cpdef cnp.ndarray[cnp.float64_t, ndim=1] grad_array(
        self,
        cnp.ndarray[cnp.float64_t, ndim=1] y_true,
        cnp.ndarray[cnp.float64_t, ndim=1] y_pred,
    ):
        if self._grad_fn is None:
            raise NotImplementedError(
                f"Loss '{self.name}' was created without a grad_fn."
            )
        cdef cnp.ndarray[cnp.float64_t, ndim=1] result	
        if self._vectorized:
            result = np.asarray(self._grad_fn(y_true, y_pred), dtype=np.float64)
            return result
        return LossFunction.grad_array(self, y_true, y_pred)

#BUILTIN NON-DEFAULT LOSSES
cdef class MAELoss(LossFunction):
    """Mean Absolute Error: |y_pred - y_true|, grad = sign(y_pred - y_true)."""

    def __init__(self):
        super().__init__(name="mae", has_grad=True)

    cpdef double evaluate_scalar(self, double y_true, double y_pred):
        cdef double r = y_pred - y_true
        return r if r >= 0.0 else -r

    cpdef double grad_scalar(self, double y_true, double y_pred):
        cdef double r = y_pred - y_true
        if r > 0.0:   return  1.0
        if r < 0.0:   return -1.0
        return 0.0


cdef class HuberLoss(LossFunction):
    """
    Huber loss with threshold delta.
      |r| <= delta : 0.5 * r^2
      |r| >  delta : delta * (|r| - 0.5*delta)
    """


    def __init__(self, double delta=1.0):
        super().__init__(name="huber", has_grad=True)
        if delta <= 0.0:
            raise ValueError("delta must be positive.")
        self.delta = delta

    cpdef double evaluate_scalar(self, double y_true, double y_pred):
        cdef double r = y_pred - y_true
        cdef double a = r if r >= 0.0 else -r
        if a <= self.delta:
            return 0.5 * r * r
        return self.delta * (a - 0.5 * self.delta)

    cpdef double grad_scalar(self, double y_true, double y_pred):
        cdef double r = y_pred - y_true
        cdef double a = r if r >= 0.0 else -r
        if a <= self.delta:
            return r
        return self.delta if r > 0.0 else -self.delta

    def __repr__(self):
        return f"HuberLoss(delta={self.delta})"


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

cdef cnp.ndarray[cnp.float64_t, ndim=1] _as_f64(arr):
    """Convert list/array to a contiguous float64 memoryview."""
    return np.ascontiguousarray(arr, dtype=np.float64)
