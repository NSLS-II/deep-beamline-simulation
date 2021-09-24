========================
Deep Beamline Simulation
========================

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. image :: https://github.com/jennmald/deep-beamline-simulation/workflows/badge.svg?branch=master
   :target: http://github.com/jennmald/deep-beamline-simulation/actions


Beamline Simulation using `sirepo-bluesky`_ and Synchrotron Radiation Workshop (`SRW`_).

* Free software: 3-clause BSD license

Purpose
-------

Use machine learning to simulate beamlines similar to SRW with lower computational costs.
The goal is to determine whether or not a neural network can construct a beamline simulation given data from sirepo-bluesky.

Installation
------------

- Install `VirtualBox`_ on your computer.
- Install `Vagrant`_ using the terminal.
- Git clone this repository. Then use ``cd deep-beamline-simulation`` to move into the top level directory of this repository.
- You will see a Vagrantfile containing virtual machine setup information. Start the virtual machine using ``vagrant up`` followed by ``vagrant ssh``.
- Once the virtual machine is running, finish the Miniconda installation,
.. code:: bash

   bash Miniconda3-latest-Linux-x86_64.sh

- Close and reload the VM. Create a conda environment, 
.. code:: bash

   conda create -n sirepo_bluesky python=3.8
   conda activate sirepo_bluesky

- It is recommended to check the status of Mongo DB using ``sudo systemctl status mongod``. If the status is 'dead' use ``sudo systemctl start mongod.service`` to start running Mongo DB.

- To view the contents of ``deep-beamline-simulation`` repository use ``cd /vagrant``. Use ``pip install .`` to install all requirements and setup necessary packages. 

- Use the command ``bash start_docker.sh`` to start the docker container. To run the docker container in the background use ``-d`` at the end of the command or for an interactive version add ``-it``. To verify the container is running use ``docker ps -a``. If you chose to shutdown the container use ``docker stop <name of container>``. In our case the docker container is called 'sirepo'.

- Open the interactive website http://10.10.10.10:8000/srw.

Pytorch
-------

- To run files requiring pytorch, visit ``https://pytorch.org`` to find a compatible installation command. For this virtual machine it is suggested to use,
.. code:: bash

   conda install pytorch torchvision torchaudio cpuonly -c pytorch

- When neural network training is complete, exit the VM and run ``python tensorfile.py``. Then use ``tensorboard --logdir=runs``. This will provide output similar to the following. Copy and paste the link into the web browser to access tensorboard.

.. code:: bash

   Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
   TensorBoard 2.5.0 at http://localhost:6006/ (Press CTRL+C to quit)

.. _sirepo-bluesky: https://github.com/NSLS-II/sirepo-bluesky
.. _SRW: https://github.com/ochubar/SRW
.. _VirtualBox: https://www.virtualbox.org/
.. _Vagrant: https://www.vagrantup.com
