from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(["real_functionC.pyx", "function_treeC.pyx", "gp_movesC.pyx"]),
    include_dirs=[np.get_include()]
)