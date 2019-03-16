#!/bin/bash
#COBALT -n 5
#COBALT -t 8:00:00
#COBALT -A datascience                           


echo "Running Cobalt Job $COBALT_JOBID."

source activate /projects/datascience/pbalapra/cooley-soft/deephyper-cooley/

mpirun -np 10 -ppn 2 python script.py

source deactivate



