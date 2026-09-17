from cpython.ref cimport Py_INCREF
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer
from real_functionC cimport RealFunctionC
from cpython.ref cimport Py_INCREF, Py_DECREF

cdef void* python_object_to_void_ptr(object py_obj):
    """From python object to void* pointer"""
    Py_INCREF(py_obj) 
    return <void*> py_obj 

cdef object void_ptr_to_python_object(void* ptr):
    """from void* pointer back to Python object"""
    return <object> ptr  

cdef RealFunctionC void_ptr_to_rf(void* ptr):
    return <RealFunctionC> ptr
