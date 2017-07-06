# ebm-analytical

## Source code for "Ice caps and ice belts: the effects of obliquity on ice-albedo feedback" by Rose, Cronin and Bitz (2017, Astrophys. J.)

### [Brian E. J. Rose](http://www.atmos.albany.edu/facstaff/brose/index.html), Department of Atmospheric and Environmental Sciences, University at Albany

The included Jupyter notebook `Ice Caps and Ice Belts -- figures.ipynb` reproduces all figures in the paper. 
Some of the figures are plotted from pre-computed datasets (included in this repository). 
All the source code used to generate these datasets is included.

The module `ebm_analytical.py` contains all the code to implement the special-functions solution of the non-dimensional EBM as described in the paper.

The code is freely available under the MIT license.

### Dependencies

- `python 2.7`
- `numpy`
- `scipy`
- `mpmath` (for complex special functions in the analytical model)
- `climlab` (for insolation and implementation of the seasonal EBM)
- `ipyparallel` (for parallelization of the seasonal model parameter sweep on a compute cluster)

Note `ipyparallel` is only needed to reproduce the laborious numerical parameter sweep of the seasonal model. The figures notebook reproduces Figures 9 and 10 of the paper from pre-computed data without this requirement.
