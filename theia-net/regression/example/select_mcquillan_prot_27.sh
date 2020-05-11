#!/bin/bash

#SBATCH -p cca
#SBATCH -J job
#SBATCH -o <path to save output file>/output_select_mcquillan_prot_27.%j
#SBATCH -e <path to save output file>/error_select_mcquillan_prot_27.%j
#SBATCH -t 00-01:00:00

DIR="<path to save models>/ts/mcquillan/prot_27/run0"

python3 ./run/select.py 0 "<path to save models>/ts/mcquillan/prot_27/run0" \
                          "prot_27" "27"