# this is not so much a script as a recipe for updating imports
mv NSLS-II-CSX-1-beamline-rsOptExport.py NSLS-II-CSX-1-beamline-rsOptExport.py.original
cat NSLS-II-CSX-1-beamline-rsOptExport.py.original | sed 's/import \(srw.*\)/from srwpy import \1/' | sed 's/import \(uti_plot_com\)/from srwpy import \1/' > NSLS-II-CSX-1-beamline-rsOptExport.py
