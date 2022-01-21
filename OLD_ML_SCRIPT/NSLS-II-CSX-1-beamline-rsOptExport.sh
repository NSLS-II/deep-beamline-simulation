#!/bin/bash
if [ -d "NSLS-II-CSX-1-beamline-rsOptExport_scan" ] && [ -n "$(ls -A NSLS-II-CSX-1-beamline-rsOptExport_scan)" ]; then
    for f in 'NSLS-II-CSX-1-beamline-rsOptExport.py' run_parallel_python.py; do
        rm -f "NSLS-II-CSX-1-beamline-rsOptExport_scan/$f"
    done
fi
run_rsopt="rsopt sample configuration 'NSLS-II-CSX-1-beamline-rsOptExport.yml'"
echo Running $run_rsopt...
eval "$run_rsopt" > "NSLS-II-CSX-1-beamline-rsOptExport.out" 2>&1
if [ $? -eq 0 ]
then
    run_py="python 'NSLS-II-CSX-1-beamline-rsOptExport.py' rsopt_run 'NSLS-II-CSX-1-beamline-rsOptExport.npy'"
    read -p "Continue? (** will run SRW 10 times **)? (Y/N): " confirm
    if [[ $confirm == [yY] ]]; then
        echo Running SRW...
        eval "$run_py"
        if [ $? -eq 0 ]
        then
            echo Propagated single electron intensities are in datasets/beam_intensities.npy
            echo Corresponding parameter values are in datasets/parameters.npy
        else
            echo "ERROR: SRW failed with return code $?"
        fi
    else
        echo "NOTE: to run SRW with generated data, use
        $run_py"
    fi
else
    echo "ERROR: rsopt failed with return code $?, see NSLS-II-CSX-1-beamline-rsOptExport.out"
    exit 99
fi
