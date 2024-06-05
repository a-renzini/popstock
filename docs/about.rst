==============================================
About ``popstock``
==============================================

:code:`popstock`: a python--based open--source package for the rapid computation of gravitational-wave background (GWB) fractional energy density spectra such as :math: `\Omega_{\rm GW}`, for a given realization of events, and its ensemble average :math: `\bar{\Omega}_{\rm GW}` for a given set of hyper-parameters  :math: `\Lambda`. 

Other than the standard python scientific libraries :code:`numpy` and :code:`scipy`, the main dependencies of the :code:`popstock` package are: 

* :code:`astropy`: a core python library used by astronomers
* :code:`bilby`: one of the most popular bayesian inference libraries for GW astronmy
* :code:`gwpopulation`: a collection of parametric binary black hole mass, redshift, and spin population models.

The :code:`popstock` package also relies on :code:`multiprocessing` (included in most python distributions) to parallelize the computations for the large number of samples required for high-precision GWB evaluation.

The gravitational-wave waveforms required to compute :math: `\Omega_{\rm GW}` are imported by :code:`bilby` from the LIGO Scientific Collaboration Algorithm Library (LAL), which is also required during package installation.
