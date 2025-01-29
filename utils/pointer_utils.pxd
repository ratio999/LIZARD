from real_functionC cimport RealFunctionC

cdef void* python_object_to_void_ptr(object py_obj)

cdef object void_ptr_to_python_object(void* ptr)

cdef RealFunctionC void_ptr_to_rf(void* ptr)
