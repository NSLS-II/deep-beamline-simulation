#!/bin/bash

month=$(date +"%m")
day=$(date +"%d")
year=$(date +"%Y")

today="./../tmp/data/${year}/${month}/${day}"

if [ -d "./../tmp/data/${year}/${month}/${day}" ]
then
    echo "Directory /path/to/dir exists."
else
    echo "Creating Directory"
    mkdir "${today}"
fi

pip install sirepo-bluesky
git clone https://github.com/NSLS-II/sirepo-bluesky.git
cd sirepo-bluesky/
git checkout tags/v0.2.0

docker run --rm -e SIREPO_AUTH_METHODS=bluesky:guest -e SIREPO_AUTH_BLUESKY_SECRET=bluesky -e SIREPO_SRDB_ROOT=/sirepo -e SIREPO_COOKIE_IS_SECURE=false -p 8000:8000 -v $PWD/sirepo_bluesky/tests/SIREPO_SRDB_ROOT:/SIREPO_SRDB_ROOT:ro,z radiasoft/sirepo:20200220.135917 bash -l -c "mkdir -v -p /sirepo/user/ && cp -Rv /SIREPO_SRDB_ROOT/* /sirepo/user/ && sirepo service http"&
