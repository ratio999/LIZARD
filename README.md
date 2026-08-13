![](img/LIZARD_CODICE.png)

## Installation

### 1. Clone

```bash
git clone -b master https://git.ligo.org/giuseppe.troian/machine-scientist.git
cd machine-scientist
```

### 2. Create the environment

The repository ships an `environment.yml` for conda. It pins Python 3.9 and the
stack the kernel needs (NumPy, SciPy, SymPy, Cython, mpi4py):

```bash
conda env create --file environment.yml
conda activate mcmc_koopman
```

The file already names the environment `mcmc_koopman`, so `-n MYENV` is only
needed if you want a different name. It is also an over-grown export (129
packages, most of them incidental), so expect a slow solve; `mamba` helps.

**Linux / WSL.** mpi4py needs a working MPI installation. Conda provides one,
but check that the `mpiexec` on your PATH is the environment's rather than the
system's:

```bash
which mpiexec        # should point inside the env, not /usr/bin
```

### 3. Build the Cython extensions

```bash
python setup.py build_ext --inplace
```


Cythonisation is extremely verbose; that is normal, and what matters is that it
ends without an error. Re-run it after editing any `.pyx` or `.pxd`: the
compiled `.so` files are what gets imported, not the sources.

## Testing

The maintained suites live in the repository root and run directly:

```bash
python test_phaseFX.py          # transparent atoms
python test_phaseMD.py          # multi-dataset joint evidence
python test_phasePP.py          # pinned parameters
```

Suites that spawn MPI workers must be launched through the environment's
`mpiexec`, or the spawn resolves to the system one and hangs:

```bash
mpiexec -n 1 python test_phase212_wire.py
mpiexec -n 1 python test_phase44.py
mpiexec -n 1 python test_phasePP.py --mpi
```

The files under `tests/` are historical.


