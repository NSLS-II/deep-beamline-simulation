Vagrant.configure("2") do |config|
  config.vm.box = "bento/ubuntu-20.04"
  config.vm.box_check_update = true

  config.vm.network "private_network", ip: "10.10.10.10"
  config.vm.network "forwarded_port", guest: 8000, host: 8000, host_ip: "127.0.0.1"

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
    apt install -y python3-pip
    # install X11 for matplotlib
    apt install -y xserver-xorg-core x11-utils

    # install docker
    apt install apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable"
    apt update
    apt install -y docker-ce
    # add the vagrant account to the docker group
    # this way the vagrant account can run docker without sudo
    usermod -aG docker vagrant

    # download miniconda to the vagrant account's home directory
    wget -P /home/vagrant https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chown vagrant:vagrant /home/vagrant/Miniconda3-latest-Linux-x86_64.sh

    # install mongodb
    wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
    echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
    apt update
    apt install -y mongodb-org
    systemctl start mongod
    systemctl enable mongod

    # install local.yml
    # this will not be needed if we can use an entrypoint
    mkdir -p /home/vagrant/.local/share/intake
    chown -Rv vagrant:vagrant /home/vagrant/.local
    cp /vagrant/local.yml /home/vagrant/.local/share/intake

    # create this directory now or it will be created by the sirepo
    # docker container with root ownership
    mkdir -p /home/vagrant/sirepo_srdb_root
    chown vagrant:vagrant /home/vagrant/sirepo_srdb_root
  SHELL
  # ssh into the VM and run the Sirepo docker container like this:
  #   mkdir -p .local/share/intake
  #
  #   mkdir -p $HOME/tmp/sirepo-docker-run
  #
  #   try this, it may not help
  #     mkdir sirepo_srdb_root
  #   if you have permission problems with directory sirepo_srdb_root
  #     sudo chown vagrant:vagrant sirepo_srdb_root
  #
  #   docker run -it --rm -e SIREPO_AUTH_METHODS=bluesky:guest -e SIREPO_AUTH_BLUESKY_SECRET=bluesky -e SIREPO_SRDB_ROOT=/sirepo -e SIREPO_COOKIE_IS_SECURE=false -p 8000:8000 -v $HOME/sirepo_srdb_root:/sirepo radiasoft/sirepo:20200220.135917 bash -l -c "sirepo service http"
  #
  # from the host go to http://10.10.10.10:8000
end

