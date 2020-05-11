#!/bin/bash

RUN=0
SAMPLE="balanced_classification"
DATA="ts"

# hyperparameter file id
HYPER=97
# number of hyperparam combinations
NHYPER=144

declare -a PARAMS=("evolve")

for n in "${!PARAMS[@]}"
do
   
echo "${PARAMS[$n]}"

# make data.sh
cat > data_${SAMPLE}_${PARAMS[$n]}.sh <<EOF
#!/bin/bash

#SBATCH -p cca
#SBATCH -J job
#SBATCH -o <path to save output file>/output_data_${SAMPLE}_${PARAMS[$n]}.%j
#SBATCH -e <path to save output file>/error_data_${SAMPLE}_${PARAMS[$n]}.%j
#SBATCH -t 00-00:30:00

CODE_DIR="<path to code directory>"
HOME_DIR="<path to home directory>"

python3 ../data.py "${RUN}" "${CODE_DIR}" "${HOME_DIR}" "${SAMPLE}" \
                   "${PARAMS[$n]}" "${DATA}"
sleep 30s

#END

EOF


# make disbatch
foo=1
NHYPER="$(($NHYPER-$foo))"

DIR="<path to save models>/${DATA}/${SAMPLE}/${PARAMS[$n]}/run${RUN}"

for i in $(seq 0 $NHYPER)
do

cat >> disbatch_${SAMPLE}_${param} <<EOF
python3 ./run/main.py $RUN "${DIR}" "${PARAMS[$n]}" "${HYPER}" $i
EOF
done


# make choose
cat >> choose_${SAMPLE}_${PARAMS[$n]}.sh <<EOF
#!/bin/bash

#SBATCH -p cca
#SBATCH -J job
#SBATCH -o <path to save output file>/output_choose_${SAMPLE}_${param}.%j
#SBATCH -e <path to save output file>/error_choose_${SAMPLE}_${param}.%j
#SBATCH -t 00-01:00:00

DIR="<path to save models>/${DATA}/${SAMPLE}/${PARAMS[$n]}/run${RUN}"

python3 ./run/select.py $RUN "${DIR}" "${PARAMS[$n]}" "${HYPER}"

EOF

done