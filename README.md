GDMA MPFIT Program in Python
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/shehan807/pyMPFit/workflows/CI/badge.svg)](https://github.com/shehan807/pyMPFit/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/shehan807/pyMPFit/branch/main/graph/badge.svg)](https://codecov.io/gh/shehan807/pyMPFit/branch/main)
[![Documentation Status](https://readthedocs.org/projects/pyMPFit/badge/?version=latest)](https://pyMPFit.readthedocs.io/en/latest/?badge=latest)

PyMPFit aims to provide tools (cf. openff-recharge) for fitting and training partial charges of molecules against quantum chemical Gaussian distributed multipole analysis (GDMA) data.

## Features (under development, following [openff-recharge](https://github.com/openforcefield/openff-recharge/tree/main) functionality)

* [ ] **Generating QC GDMA multipole moment data**
	* [ ] directly interfacing with the [Psi4 GDMA]((https://psicode.org/psi4manual/master/gdma.html) quantum chemical code 
	* [ ] from wavefunctions stored within QCFractal instance, including the [QCArchive](https://qcarchive.molssi.org/)
* [ ] **Defining new charge models that contain**
	* [ ] virtual sites 
* [ ] **A SMARTS port of the GDMA MPFIT charge model**


### Copyright

Copyright (c) 2025, Shehan Parmar


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.10.
