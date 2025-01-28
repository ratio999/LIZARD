from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os

from distutils.extension import Extension



setup(
        ext_modules=cythonize(["real_functionC.pyx", 
                    "function_treeC.pyx", "gp_movesC.pyx", 
                    "groveC.pyx", "samplerC.pyx", "num_wrapC.pyx"
                           ], 
                          compiler_directives={'cdivision': True, 'boundscheck': True}, gdb_debug=True
),
    include_dirs=[np.get_include()]
)
