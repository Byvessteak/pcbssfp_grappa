Acceleration and GRAPPA reconstruction for phase-cycled bSSFP MRI signals

MATLAB-FILES:
- create_plots.m			to create plots (T1/T2/bSSFP profile) of reconstructed images
- undersample_data_5D.m	to retrospectivelly undersample multi-slice PC-bSSFP data

PYTHON-FILES:
- RUN_GRAPPA_Matlab.py				to run single-slice GRAPPA for undersampled k-space .mat files
- RUN_GRAPPA_Matlab_5D_multislice_con.py	to run multi-slice GRAPPA for undersampled k-space .mat files
- RUN_k_t_Grappa_Matlab.py				to run k-t GRAPPA for .mat files


To run the python files, the pygrappa library is needed:
https://github.com/mckib2/pygrappa

To run the Matlab files, further non-published methods are needed.
