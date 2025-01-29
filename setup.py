from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os

from distutils.extension import Extension






extensions = [
    Extension("real_functionC", sources=["real_functionC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("function_treeC", sources=["function_treeC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("gp_movesC", sources=["gp_movesC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("groveC", sources=["groveC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("samplerC", sources=["samplerC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("num_wrapC", sources=["num_wrapC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    
    
    Extension("utils.pointer_utils", 
              sources=["utils/pointer_utils.pyx"], 
              include_dirs=[np.get_include(), os.path.abspath(".")],
              
              library_dirs=[os.path.join(os.path.abspath("."), "utils")],
              libraries=[])  
]


setup(
    ext_modules=cythonize(extensions, compiler_directives={'cdivision': True, 'boundscheck': True}, gdb_debug=True),
    include_dirs=[np.get_include(), os.path.abspath(".")],
)
