![](img/LIZARD_CODICE.png)

## Installation

### 1. Clone

```bash
git clone -b release https://git.ligo.org/giuseppe.troian/machine-scientist.git
cd machine-scientist
```

The repository is laid out as follows:

```
src/           the Cython kernel (.pyx/.pxd) and the Python modules the
               tests and experiments share (exp_common, exp_constraints,
               i3b_pool, i3b_polya_pricing, feynman_exactness)
tests/         the maintained test suites
experiments/   experiment drivers, plotting scripts and launch wrappers;
               they write their outputs to experiments/results/
benchmarks/    speed benchmarks
```

Modules keep flat names (`import groveC`). The tests, experiment drivers and
benchmarks put `src/` on `sys.path` themselves, so no installation step is
needed.

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

Run it from the repository root:

```bash
python setup.py build_ext --inplace
```


Cythonisation is extremely verbose; that is normal, and what matters is that it
ends without an error. The compiled `.so` files land in `src/`, next to their
sources. Re-run the build after editing any `.pyx` or `.pxd`: the compiled
files are what gets imported, not the sources.

## Testing

The maintained suites live in `tests/` and run directly from the repository
root. Use one BLAS thread; for bit-reproducible MPI runs also pin the hash
seed:

```bash
export OMP_NUM_THREADS=1 PYTHONHASHSEED=0
python tests/test_phaseFX.py          # transparent atoms
python tests/test_phaseMD.py          # multi-dataset joint evidence
python tests/test_phasePP.py          # pinned parameters
```

Suites that spawn MPI workers must be launched through the environment's
`mpiexec`, or the spawn resolves to the system one and hangs:

```bash
mpiexec -n 1 python tests/test_phase212_wire.py
mpiexec -n 1 python tests/test_phase44.py
mpiexec -n 1 python tests/test_phasePP.py --mpi
```

`tests/legacy/` holds an old smoke test that is not maintained.


