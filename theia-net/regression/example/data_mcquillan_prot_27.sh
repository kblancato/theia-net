#!/bin/bash

#SBATCH -p cca
#SBATCH -J job
#SBATCH -o <path to save output file>/output_data_mcquillan_prot_27.%j
#SBATCH -e <path to save output file>/error_data_mcquillan_prot_27.%j
#SBATCH -t 00-00:30:00

python3 ./run/data.py "0" "<path to code directory>" "<path to home directory>"\
                      "mcquillan" "prot_27" "ts"
sleep 30s

#END

