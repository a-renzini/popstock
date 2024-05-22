==============================================
About ``popstock``
==============================================

`\tt popstock`: a python--based open--source package for the rapid computation of gravitational-wave background (GWB) fractional energy density spectra such as :math: `\Omega_{\rm GW}`, for a given realization of events, and its ensemble average :math: `\bar{\Omega}_{\rm GW}` for a given set of hyper-parameters  :math: `\Lambda`. 

Other than the standard python scientific libraries `numpy` and `scipy`, the main dependencies of the `popstock` package are: 

* `astropy`: a core python library used by astronomers
* `bilby`: one of the most popular bayesian inference libraries for GW astronmy
* `gwpopulation`: a collection of parametric binary black hole mass, redshift, and spin population models.

The `popstock` package also relies on `multiprocessing` (included in most python distributions) to parallelize the computations for the large number of samples required for high-precision GWB evaluation.

The gravitational-wave waveforms required to compute `\Omega_{\rm GW}` are imported by `bilby` from the LIGO Scientific Collaboration Algorithm Library (LAL), which is also required during package installation.
