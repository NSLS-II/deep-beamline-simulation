========================
Deep Beamline Simulation
========================

.. image:: https://img.shields.io/travis/jennmald/deep-beamline-simulation.svg
        :target: https://travis-ci.org/jennmald/deep-beamline-simulation

.. image:: https://img.shields.io/pypi/v/deep-beamline-simulation.svg
        :target: https://pypi.python.org/pypi/deep-beamline-simulation


Beamline Simulation using `sirepo-bluesky`_ and Synchrotron Radiation Workshop (`SRW`_).

* Free software: 3-clause BSD license

Purpose
-------

Use machine learning to simulate beamlines similar to SRW with lower computational costs.

Installation
------------

- Install `VirtualBox`_ on your computer
- Install `Vagrant`_ using the terminal
- Add the Vagrantfile located in this repository to a new directory
- Start the virtual machine using ``vagrant up`` followed by ``vagrant ssh``
- Once in the VM finish the Miniconda installation
.. code:: bash

   bash Miniconda3-latest-Linux-x86_64.sh

- Create a conda environment 
.. code:: bash

   conda create -n sirepo_bluesky python=3.8
   conda activate sirepo_bluesky
   sudo apt update
   sudo apt upgrade

- Install sirepo-bluesky
.. code:: bash
   pip install sirepo-bluesky
   git clone https://github.com/NSLS-II/sirepo-bluesky/
   cd sirepo_bluesky/

- Create a directory ``mkdir /home/temp/sirepo-docker-run`` and run the docker container
..code:: bash
  docker run -it --rm -e SIREPO_AUTH_METHODS=bluesky:guest -e SIREPO_AUTH_BLUESKY_SECRET=bluesky -e SIREPO_SRDB_ROOT=/sirepo -e SIREPO_COOKIE_IS_SECURE=false -p 8000:8000 -v $HOME/sirepo_srdb_root:/sirepo radiasoft/sirepo:20200220.135917 bash -l -c "sirepo service http"

- Open a new terminal window, ``vagrant up``, ``vagrant ssh``, activate the conda environment,
and enter the directory for ``sirepo-bluesky``. Run ``ipython`` to begin simulations.

.. _sirepo-bluesky: https://github.com/NSLS-II/sirepo-bluesky
.. _SRW: https://www.esrf.fr/Accelerators/Groups/InsertionDevices/Software/SRW
.. _VirtualBox: https://www.virtualbox.org/
.. _Vagrant: https://www.vagrantup.com
