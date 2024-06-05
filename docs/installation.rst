.. _installation:

============
Installation
============

.. _installing-popstock:

Installing from source
======================

These are instructions to install :code:`popstock`, which runs on Python :math:`\ge3.9` and above.

Currently, there is one beta PyPI release (see the `pygwb PyPi page <https://pypi.org/project/popstock/>`_ for more details):

.. code-block:: console

   0.0.1

which may be installed using :code:`pip`:

.. code-block:: console

   $ pip install popstock==[version]

Otherwise, you may install the cloned repository directly. If you already have an existing Python environment, you can simply clone the code and install in any of the usual ways.

.. tabs::

  .. tab:: pypi

    .. code-block:: console

      $ git clone git@github.com:a-renzini/popstock.git
      $ pip install .

  .. tab:: setup.py

    .. code-block:: console

      $ git clone git@github.com:a-renzini/popstock.git
      $ python setup.py install

You may also wish to install in "develop" mode.

.. tabs::

  .. tab:: pypi

    .. code-block:: console

      $ git clone git@github.com:a-renzini/popstock.git
      $ pip install -e .[dev]

  .. tab:: setup.py

    .. code-block:: console

      $ git clone git@github.com:a-renzini/popstock.git
      $ python setup.py develop

In develop mode, a symbolic link is made between the source directory and the environment site packages.
This means that any changes to the source are immediately propagated to the environment.

.. _creating-environment:

Creating a python environment
=============================

We recommend working with a recent version of Python.
A good reference is to use the default anaconda version.
This is currently :code:`Python 3.9` (October 2020).

The general recommendation when getting started is to create a fresh conda environement with a known
version of Python and installing :code:`popstock` as specified above. All required dependencies 
will automatically be installed, and this should minimise dependency conflicts. 

.. tabs::

   .. tab:: conda

      :code:`conda` is a recommended package manager which allows you to manage
      installation and maintenance of various packages in environments. For
      help getting started, see the `LSCSoft documentation <https://lscsoft.docs.ligo.org/conda/>`_.

      For detailed help on creating and managing environments see `these help pages
      <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
      Here is an example of creating and activating an environment named pygwb

      .. code-block:: console

         $ conda create -n popstock_env python=3.9
         $ conda activate popstock_env

   .. tab:: virtualenv

      :code:`virtualenv` is a similar tool to conda. To obtain an environment, run

      .. code-block:: console

         $ virtualenv --python=/usr/bin/python3.9 $HOME/virtualenvs/popstock_env
         $ source virtualenvs/popstock_env/bin/activate


.. note::
  Please report broken versions and dependencies as soon as possible by opening an `issue <https://github.com/a-renzini/popstock/issues>`_!
