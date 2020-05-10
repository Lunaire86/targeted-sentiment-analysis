#!/bin/bash

#SBATCH --job-name=in5550
#SBATCH --account=nn9447k

# Time limit for the job. Lower limit means higher job priority.
#SBATCH --time=03:00:00

# Number of compute nodes for the job. Default is 1.
#SBATCH --nodes=1

# Per core memory limit. 3000MB by default (?)
#SBATCH --mem-per-cpu=12G

# Number of cores on the compute node. Request two by default,
# or between 6 and 10 for larger computations.
#SBATCH --ntasks-per-node=2

source ${HOME}/.bashrc

# when running under SLURM control, i.e. as an actual batch job, box in NumPy
# (assuming we stick to the OpenBLAS back-end) to respect our actual allocation
# of cores.
#
if [ -n "${SLURM_JOB_NODELIST}" ]; then
  export OPENBLAS_NUM_THREADS=${SLURM_CPUS_ON_NODE}
fi

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load IN5550
module purge
module use -a /cluster/shared/nlpl/software/modules/etc
module add nlpl-in5550/202005/3.7
module add nlpl-tensorflow/2.0.0/3.7
module add nlpl-fasttext/0.9.2/3.7

echo "submission directory: ${SUBMITDIR}"
ulimit -a module list

# pass on the remaining command-line options
python babil "${@}"