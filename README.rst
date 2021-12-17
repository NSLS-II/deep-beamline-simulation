========================
Deep Beamline Simulation
========================

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. image:: https://github.com/jennmald/deep-beamline-simulation/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/jennmald/deep-beamline-simulation/actions/workflows/tests.yml

Beamline Simulation using `sirepo-bluesky`_ and Synchrotron Radiation Workshop (`SRW`_).

* Free software: 3-clause BSD license

Purpose
-------
Use machine learning techniques to simulate beamlines given data from SRW.

Installation
------------

- Install `VirtualBox`_ on your computer.
- Install `Vagrant`_ using the terminal.
- Git clone this repository. Then use ``cd deep-beamline-simulation`` to move into the top level directory of this repository.
- You will see a Vagrantfile containing virtual machine setup information. Start the virtual machine using ``vagrant up`` followed by ``vagrant ssh``. To reload the virtual machine use ``vagrant reload`` and to update the machine for any changes use ``vagrant provision``.

- It is recommended to check the status of Mongo DB using ``sudo systemctl status mongod``. If the status is 'dead' use ``sudo systemctl start mongod.service`` to start running Mongo DB.

- To view the contents of ``deep-beamline-simulation`` repository use ``cd /vagrant``.

- There will be a conda environment created using the Vagrantfile. Verify this by using ``conda env list``. To activate it use ``conda activate dbs``.

- Use ``pip install .`` to install all requirements and setup necessary packages. 

- To run the docker container for Sirepo, use command ``bash scripts/start_sirepo.sh -it``. To run the container in the background use ``-d`` instead. The default ``-it`` will run the container in interactive mode. Using interactive mode will force you to open a new terminal window to view code and make changes. In the new window use ``Vagrant ssh`` to join the session created eariler and activate conda using the same command as above. 

- To verify the container is running use ``docker ps -a``. If you chose to shutdown the container use ``docker stop <name of container>``. In our case the docker container is called 'sirepo'.

- Open the interactive website `localhost`_.

Interactive Tensorboard
-----------------------

- There are a few neural networks found in this repository. Pytorch is installed in the Vagrantfile and while running the pip install. There are a few extra steps to be able to use tensorboard applications.

- When neural network training is complete, exit the virtual machine and run ``python tensorfile.py``. Then use ``tensorboard --logdir=runs``. This will provide output similar to the following. Copy and paste the link into the web browser to access tensorboard.

.. code:: bash

   Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
   TensorBoard 2.5.0 at http://localhost:6006/ (Press CTRL+C to quit)

.. _sirepo-bluesky: https://github.com/NSLS-II/sirepo-bluesky
.. _SRW: https://github.com/ochubar/SRW
.. _VirtualBox: https://www.virtualbox.org/
.. _Vagrant: https://www.vagrantup.com
.. _localhost: http://localhost:8000/en/landing.html
