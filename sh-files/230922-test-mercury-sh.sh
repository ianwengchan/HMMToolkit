#!/bin/bash

# module load apptainer

# change to home directory
cd ~/
echo "Current directory: $(pwd)"

echo $(singularity exec --home ~/ singularity/julia_1.9.sif which julia)
echo $(singularity exec --home ~/ singularity/julia_1.9.sif julia --version)

singularity exec --home ~/ singularity/julia_1.9.sif julia --optimize=3 --min-optlevel=3 -t 10 ~/cthmm/FitHMM-jl/scripts/hello-world.jl