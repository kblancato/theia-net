#!/bin/bash

#SBATCH -p cca
#SBATCH -J job
#SBATCH -o <path to save output file>/output_data_balanced_classification_evolve.%j
#SBATCH -e <path to save output file>/error_data_balanced_classification_evolve.%j
#SBATCH -t 00-00:30:00

python3 ./run/data.py "0" "<path to code directory>" "<path to home directory>"\
                      "balanced_classification" "evolve" "ts"
sleep 30s

#END

