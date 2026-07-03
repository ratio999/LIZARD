![](img/LIZARD_CODICE.png)

## Installation
### Cloning
The installation can be done cloning this repository:

```
git clone -b master https://git.ligo.org/giuseppe.troian/machine-scientist.git
cd machine-scientist
```
### Install with conda
The repository provides a `environment.yml` file that can be used for conda installation.
This file still contains a lot of trash and lists way too many packages.


```
conda env create --file environment.yml -n MYENV
```

###Try removing the `-n MYENV` if it does not work.


## Setup

For setting up the cython code anc compiling C modules:

```
python setup.py build_ext --inplace
```


The whole cythonization and compilation of files is very verbose, though no error message should be displayed.  
## Testing
The only test actively supported and modified so far is `tests/test_cython3.py`.
Tests have to be moved from the `tests/` folder into the main one, and run with:
```
python test_cython3.py
```


