from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["real_functionC.pyx", "function_treeC.pyx", "gp_movesC.pyx", "samplerC.pyx",
                            "fitnessC.pyx", "posteriorC.pyx", "gsl_optimizerC.pyx"], 
                          compiler_directives={'cdivision': True, 'boundscheck': True}, gdb_debug=True),
    include_dirs=[np.get_include()]
)