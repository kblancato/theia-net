#!/bin/bash

#SBATCH -p cca
#SBATCH -J job
#SBATCH -o <path to save output file>/output_choose_balanced_classification_evolve.%j
#SBATCH -e <path to save output file>/error_choose_balanced_classification_evolve.%j
#SBATCH -t 00-01:00:00

DIR="<path to save models>/ts/balanced_classification/evolve/run0"

python3 ./run/select.py 0 "<path to save models>/ts/balanced_classification/evolve/run0" \
                           "evolve" "97"

