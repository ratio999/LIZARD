from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os

from distutils.extension import Extension

try:
    import mpi4py
    mpi_include = mpi4py.get_include()
except ImportError:
    mpi_include = None

# For the actual MPI library headers, you may also need:
# On Linux: usually /usr/include/mpi or from mpicc --showme:compile
# On Windows with MS-MPI: typically C:\Program Files (x86)\Microsoft SDKs\MPI\Include

# Try to get MPI include from mpicc (Linux/WSL)
import subprocess
try:
    result = subprocess.run(['mpicc', '--showme:compile'], capture_output=True, text=True)
    mpi_compile_flags = result.stdout.strip().split()
    mpi_include_dirs = [f[2:] for f in mpi_compile_flags if f.startswith('-I')]
except:
    mpi_include_dirs = []

# Windows MS-MPI paths
if os.name == 'nt':
    ms_mpi_include = os.environ.get('MSMPI_INC', r'C:\Program Files (x86)\Microsoft SDKs\MPI\Include')
    if os.path.exists(ms_mpi_include):
        mpi_include_dirs.append(ms_mpi_include)

# Combine all includes for groveC
grove_includes = [np.get_include(), os.path.abspath(".")]
if mpi_include:
    grove_includes.append(mpi_include)
grove_includes.extend(mpi_include_dirs)




extensions = [
    Extension("real_functionC", sources=["real_functionC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("function_treeC", sources=["function_treeC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("gp_movesC", sources=["gp_movesC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("groveC", sources=["groveC.pyx"], include_dirs=grove_includes),
    #Extension("samplerC", sources=["samplerC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("num_wrapC", sources=["num_wrapC.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    Extension("mpi_worker", sources=["mpi_worker.pyx"], include_dirs=[np.get_include(), os.path.abspath(".")]),
    
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
