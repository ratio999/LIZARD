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


SRC = os.path.abspath("src")


grove_includes = [np.get_include(), SRC]
if mpi_include:
    grove_includes.append(mpi_include)
grove_includes.extend(mpi_include_dirs)




extensions = [
    Extension("real_functionC", sources=["src/real_functionC.pyx"], include_dirs=[np.get_include(), SRC]),
    Extension("function_treeC", sources=["src/function_treeC.pyx"], include_dirs=[np.get_include(), SRC]),
    Extension("gp_movesC", sources=["src/gp_movesC.pyx"], include_dirs=[np.get_include(), SRC]),
    Extension("polya_treeC", sources=["src/polya_treeC.pyx"], include_dirs=[np.get_include(), SRC]),
    Extension("groveC", sources=["src/groveC.pyx"], include_dirs=grove_includes),
    Extension("mpi_worker", sources=["src/mpi_worker.pyx"], include_dirs=[np.get_include(), SRC]),
    Extension("loss_function", sources=["src/loss_function.pyx"], include_dirs=[np.get_include(), SRC]),

    Extension("utils.pointer_utils",
              sources=["src/utils/pointer_utils.pyx"],
              include_dirs=[np.get_include(), SRC],

              library_dirs=[os.path.join(SRC, "utils")],
              libraries=[])
]


setup(
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, compiler_directives={'cdivision': True, 'boundscheck': True}, gdb_debug=True,
                          include_path=[SRC]),
    include_dirs=[np.get_include(), SRC],
)
