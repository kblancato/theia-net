#!/bin/bash

RUN=0
SAMPLE="mcquillan"
DATA="ts"

declare -a PARAMS=("prot_97" "prot_62" "prot_27" "prot_14")

# hyperparameter files
declare -a HYPER=("97" "62" "27" "14")
# number of hyperparam combinations
NHYPER=144

for n in "${!PARAMS[@]}"
do

echo "${PARAMS[$n]}"
echo "${HYPER[$n]}"

# make data.sh
cat > data_${SAMPLE}_${PARAMS[$n]}.sh <<EOF
#!/bin/bash

#SBATCH -p cca
#SBATCH -J job
#SBATCH -o <path to save output file>/output_data_${SAMPLE}_${PARAMS[$n]}.%j
#SBATCH -e <path to save error file>/error_data_${SAMPLE}_${PARAMS[$n]}.%j
#SBATCH -t 00-00:30:00

CODE_DIR="<path to code directory>"
HOME_DIR="<path to home directory>"

python3 ./run/data.py "${RUN}" "${CODE_DIR}" "${HOME_DIR}" "${SAMPLE}" \
                      "${PARAMS[$n]}" "${DATA}"
sleep 30s

#END

EOF


# make disbatch file
foo=1
NHYPER="$(($NHYPER-$foo))"

DIR="<path to save models>/${DATA}/${SAMPLE}/${PARAMS[$n]}/run${RUN}"

for i in $(seq 0 $NHYPER)
do

cat >> disbatch_${SAMPLE}_${PARAMS[$n]} <<EOF
python3 ./run/main.py "${RUN}" "${DIR}" "${PARAMS[$n]}" "${HYPER[$n]}" $i
EOF
done


# make select.sh
cat >> select_${SAMPLE}_${PARAMS[$n]}.sh <<EOF
#!/bin/bash

#SBATCH -p cca
#SBATCH -J job
#SBATCH -o <path to save output file>/output_select_${SAMPLE}_${PARAMS[$n]}.%j
#SBATCH -e <path to save output file>/error_select_${SAMPLE}_${PARAMS[$n]}.%j
#SBATCH -t 00-01:00:00

DIR="<path to save models>/${DATA}/${SAMPLE}/${PARAMS[$n]}/run${RUN}"

python3 ./run/select.py $RUN "${DIR}" "${PARAMS[$n]}" "${HYPER[$n]}"

EOF

done