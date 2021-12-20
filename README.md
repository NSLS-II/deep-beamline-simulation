# Deep Beamline Simulation

[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![tests](https://github.com/jennmald/deep-beamline-simulation/actions/workflows/tests.yml/badge.svg)](https://github.com/jennmald/deep-beamline-simulation/actions/workflows/tests.yml) 

Beamline Simulation using [sirepo-bluesky](https://github.com/NSLS-II/sirepo-bluesky) and [Synchrotron Radiation Workshop](https://github.com/ochubar/SRW).

* Free software: 3-clause BSD license

## Purpose

Use machine learning techniques to simulate beamlines given data from SRW.

## Installation

- Install [VirtualBox](https://www.virtualbox.org/) on your computer.

- Install [Vagrant](https://www.vagrantup.com) using the terminal.

- Git clone this repository. Then use ``cd deep-beamline-simulation`` to move into the top level directory of this repository.

- You will see a Vagrantfile containing virtual machine setup information. Start the virtual machine using ``vagrant up`` followed by ``vagrant ssh``. To reload the virtual machine use ``vagrant reload`` and to update the machine for any changes use ``vagrant provision``.

``` bash
    % vagrant up
    Bringing machine 'default' up with 'virtualbox' provider...
    ==> default: Checking if box 'bento/ubuntu-20.10' version '202107.28.0' is up to date...
    ==> default: Clearing any previously set forwarded ports...
    ==> default: Fixed port collision for 22 => 2222. Now on port 2201.
    ==> default: Machine already provisioned. Run `vagrant provision` or use the `--provision`
    ==> default: flag to force provisioning. Provisioners marked to run always will still run.
    % vagrant ssh
    Welcome to Ubuntu 20.10 (GNU/Linux 5.8.0-63-generic x86_64)
    * Documentation:  https://help.ubuntu.com
    * Management:     https://landscape.canonical.com
    * Support:        https://ubuntu.com/advantage
  vagrant@vagrant:~$
```

- It is recommended to check the status of Mongo DB using ``sudo systemctl status mongod``. If the status is 'dead' use ``sudo systemctl start mongod`` to start running Mongo DB.

``` bash 
  vagrant@vagrant:~$ sudo systemctl status mongod
  ● mongod.service - MongoDB Database Server
       Loaded: loaded (/lib/systemd/system/mongod.service; enabled; vendor preset: enabled)
       Active: inactive (dead) since Mon 2021-12-20 14:13:47 UTC; 3s ago
         Docs: https://docs.mongodb.org/manual
      Process: 647 ExecStart=/usr/bin/mongod --config /etc/mongod.conf (code=exited, status=0/SUCCESS)
     Main PID: 647 (code=exited, status=0/SUCCESS)
  Dec 20 14:07:41 vagrant systemd[1]: Started MongoDB Database Server.
  Dec 20 14:13:47 vagrant systemd[1]: Stopping MongoDB Database Server...
  Dec 20 14:13:47 vagrant systemd[1]: mongod.service: Succeeded.
  Dec 20 14:13:47 vagrant systemd[1]: Stopped MongoDB Database Server.
  vagrant@vagrant:~$ sudo systemctl start mongod
  vagrant@vagrant:~$ sudo systemctl status mongod
  ● mongod.service - MongoDB Database Server
       Loaded: loaded (/lib/systemd/system/mongod.service; enabled; vendor preset: enabled)
       Active: active (running) since Mon 2021-12-20 14:13:57 UTC; 1s ago
         Docs: https://docs.mongodb.org/manual
     Main PID: 2655 (mongod)
       Memory: 156.0M
       CGroup: /system.slice/mongod.service
               └─2655 /usr/bin/mongod --config /etc/mongod.conf
  Dec 20 14:13:57 vagrant systemd[1]: Started MongoDB Database Server.

- To view the contents of ``deep-beamline-simulation`` repository use ``cd /vagrant``.

``` bash
  (dbs) vagrant@vagrant:~$ cd /vagrant/
```

- There will be a conda environment created using the Vagrantfile. Verify this by using ``conda env list``. To activate it use ``conda activate dbs``.

``` bash
  vagrant@vagrant:~$ conda env list
   conda environments:

  base                  *  /home/vagrant/miniconda3
  dbs                      /home/vagrant/miniconda3/envs/dbs

  vagrant@vagrant:~$ conda activate dbs
```

- Use ``pip install .`` to install all requirements and setup necessary packages. 

``` bash 
  (dbs) vagrant@vagrant:/vagrant$ pip install .

  ...

  Successfully built deep-beamline-simulation
  Installing collected packages: deep-beamline-simulation, ...

  ...

  Successfully installed deep-beamline-simulation-0.post246.dev0+g97d4ced ...
```

- To run the docker container for Sirepo, use command ``bash scripts/start_sirepo.sh -it``. To run the container in the background use ``-d`` instead. The default ``-it`` will run the container in interactive mode. Using interactive mode will force you to open a new terminal window to view code and make changes. In the new window use ``vagrant ssh`` to join the session created eariler and activate conda using the same command as above. 

``` bash
  (dbs) vagrant@vagrant:/vagrant$ bash scripts/start_sirepo.sh -it
  Creating Directory /home/vagrant/tmp/data/2021/12/20
  ...
  docker.io/radiasoft/sirepo:beta
  REPOSITORY         TAG       IMAGE ID       CREATED       SIZE
  radiasoft/sirepo   beta      5becae748c04   5 days ago    5.76GB
  radiasoft/sirepo   <none>    8117306ff3a6   3 weeks ago   5.76GB
  radiasoft/sirepo   <none>    9b56b3e3a7ff   5 weeks ago   5.76GB
  Command to run:

  docker run -it --init --rm --name sirepo        -e SIREPO_AUTH_METHODS=bluesky:guest        -e SIREPO_AUTH_BLUESKY_SECRET=bluesky        -e SIREPO_SRDB_ROOT=/sirepo        -e SIREPO_COOKIE_IS_SECURE=false        -p 8000:8000        radiasoft/sirepo:beta bash -l -c "mkdir -v -p /sirepo && sirepo service http"

  ...

   * Serving Flask app 'sirepo.server' (lazy loading)
   * Environment: development
   * Debug mode: off
   * Running on all addresses.
     WARNING: This is a development server. Do not use it in a production deployment.
   * Running on http://172.17.0.2:8000/ (Press CTRL+C to quit)
   * Restarting with stat
```


- To verify the container is running use ``docker ps -a``. If you chose to shutdown the container use ``docker stop <name of container>``. In our case the docker container is called 'sirepo'.

``` bash
  vagrant@vagrant:~$ docker ps -a
  CONTAINER ID   IMAGE                   COMMAND                  CREATED         STATUS         PORTS                                       NAMES
  f0d01fee65cf   radiasoft/sirepo:beta   "bash -l -c 'mkdir -…"   9 minutes ago   Up 9 minutes   0.0.0.0:8000->8000/tcp, :::8000->8000/tcp   sirepo

  vagrant@vagrant:~$ docker stop sirepo
  sirepo
```

- Open the interactive website [localhost](http://localhost:8000/en/landing.html).

## Interactive Tensorboard

- There are a few neural networks found in this repository. Pytorch is installed in the Vagrantfile and while running the pip install. There are a few extra steps to be able to use tensorboard applications.

- When neural network training is complete, exit the virtual machine and run ``python tensorfile.py``. Then use ``tensorboard --logdir=runs``. This will provide output similar to the following. Copy and paste the link into the web browser to access tensorboard.


``` bash
   Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
   TensorBoard 2.5.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

