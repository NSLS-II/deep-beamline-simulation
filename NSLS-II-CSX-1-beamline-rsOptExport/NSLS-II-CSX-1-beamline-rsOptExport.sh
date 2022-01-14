#!/bin/bash
if [[ "$@" ]]; then
    echo usage: bash 'NSLS-II-CSX-1-beamline-rsOptExport.sh' 1>&2
    exit 0
fi
set -euo pipefail
run_dir=rsopt-srw-$(date +%Y%m%d%H%M%S)
mkdir "$run_dir"
cp 'NSLS-II-CSX-1-beamline-rsOptExport.yml' 'NSLS-II-CSX-1-beamline-rsOptExport.py' 'mirror_1d.dat'  "$run_dir"
cd "$run_dir"
run_rsopt=( rsopt sample configuration 'NSLS-II-CSX-1-beamline-rsOptExport.yml' )
echo "Running $run_rsopt"
echo "Output is in 'NSLS-II-CSX-1-beamline-rsOptExport.out'"
echo "Entering $PWD"
"${run_rsopt[@]}" > 'NSLS-II-CSX-1-beamline-rsOptExport.out' 2>&1
if [ $? -eq 0 ]
then
    run_py=( python 'NSLS-II-CSX-1-beamline-rsOptExport.py' rsopt_run 'NSLS-II-CSX-1-beamline-rsOptExport.npy' )
    read -p "Continue? (** will run SRW 100 times **)? (Y/N): " confirm
    if [[ $confirm == [yY] ]]; then
        echo Running SRW...
        if "${run_py[@]}"; then
            echo Results are in datasets/results.h5
        else
            echo "ERROR: SRW failed with return code $?"
        fi
    else
        echo "NOTE: to run SRW with generated data:
        cd $run_dir
        $run_py"
    fi
else
    echo "ERROR: rsopt failed with return code $?, see NSLS-II-CSX-1-beamline-rsOptExport.out"
    exit 99
fi
