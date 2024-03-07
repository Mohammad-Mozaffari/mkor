#!/bin/bash
#SBATCH -J bert-mkor           # Job name
#SBATCH -o bert-mkor.o%j       # Name of stdout output file
#SBATCH -e bert-mkor.e%j       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 6               # Total # of nodes 
#SBATCH -n 6              # Total # of mpi tasks
#SBATCH -t 48:00:00        # Run time (hh:mm:ss)


OPTIMIZER=lamb
PHASE=2
SINGULARITY=True

scontrol show hostname `echo $SLURM_JOB_NODELIST` > hostfile


mkdir -p sbatch_logs

HOSTFILE=hostfile
cat $HOSTFILE

MASTER_RANK=$(head -n 1 $HOSTFILE)
NNODES=$(< $HOSTFILE wc -l)
PROC_PER_NODE=3


if [[ -z "$PRUNE_INPUTS" ]]; then
    OUTPUT_DIR="results/phase${PHASE}_${OPTIMIZER}_${NNODES}nodes"
else
    OUTPUT_DIR="results/phase${PHASE}_${OPTIMIZER}_prune_inputs_${NNODES}nodes"
fi


mpirun -np $NNODES -hostfile $HOSTFILE -ppn 1 bash scripts/launch_pretraining.sh  \
        --ngpus $PROC_PER_NODE --nnodes $NNODES --master $MASTER_RANK --singularity $SINGULARITY --phase $PHASE \
	    --kwargs \
        --input_dir /tmp/data/phase${PHASE} \
        --output_dir ${OUTPUT_DIR} \
        --config_file config/bert_pretraining_phase${PHASE}_config.json \
        --weight_decay 0.01 \
        --num_steps_per_checkpoint 100 \
        --optimizer ${OPTIMIZER} \
        ${PRUNE_INPUTS} 


sleep 172800
