Vagrant.configure("2") do |config|
  config.vm.box = "bento/ubuntu-20.10"
  config.vm.box_check_update = true

  config.vm.network "private_network", ip: "10.10.10.10"
  config.vm.network "forwarded_port", guest: 8000, host: 8000, host_ip: "127.0.0.1"
  config.vm.network "forwarded_port", guest: 27017, host: 27017, host_ip: "127.0.0.1"

  config.vm.provider "virtualbox" do |vb|
    vb.gui = false
    # vb.memory = "4096"
    # vb.cpus = 4
  end

  config.ssh.forward_agent = true
  config.ssh.forward_x11 = true

  config.vm.provision "shell", inline: <<-SHELL
    # https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04
    apt update
    apt full-upgrade
    apt install -y python3-pip
    # install X11 for matplotlib
    apt install -y xserver-xorg-core x11-utils x11-apps

    # install docker
    apt install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
    apt update
    apt install -y docker-ce
    # add the vagrant account to the docker group
    # this way the vagrant account can run docker without sudo
    usermod -aG docker vagrant

    # install podman
    apt update
    apt install -y podman
    # configure default init for podman
    sed 's;# init_path = "/usr/libexec/podman/catatonit";init_path = "/usr/bin/tini";g' -i /etc/containers/containers.conf

    # install miniconda3
    wget -P /tmp https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /home/vagrant/miniconda3
    rm /tmp/Miniconda3-latest-Linux-x86_64.sh
    /home/vagrant/miniconda3/bin/conda init --system
    /home/vagrant/miniconda3/bin/conda update conda -y

    # create a conda environment for development
    /home/vagrant/miniconda3/bin/conda create -y -n dbs python=3.8
    # install deep-beamline-simulation
    /home/vagrant/miniconda3/envs/dbs/bin/pip install -e /vagrant
    /home/vagrant/miniconda3/envs/dbs/bin/pip install -r /vagrant/requirements-dev.txt
    # must change ownership for /home/vagrant/miniconda3 after creating virtual environments and installing packages
    chown -R vagrant:vagrant /home/vagrant/miniconda3

    # install mongodb
    wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
    apt update
    apt install -y mongodb-org

    # note: change the mongodb bindIP in /etc/mongod.conf to 0.0.0.0 to allow connections from the host
    cp /vagrant/files/mongod.conf /etc/

    systemctl start mongod
    systemctl enable mongod

    # databroker will look for this directory
    # it should probably be created in scripts/start_sirepo.sh
    cd /home/vagrant
    mkdir -p .local/share/intake
    chown -Rv vagrant:vagrant .local

  SHELL
  # ssh into the VM
  #   $ vagrant ssh
  # run the Sirepo docker container like this:
  #  (dbs) # bash scripts/start_sirepo.sh
  #
  # from the host go to http://127.0.0.1:8000
end
