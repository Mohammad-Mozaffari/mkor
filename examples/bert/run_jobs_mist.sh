#!/bin/bash
#SBATCH -J lamb-p1           # Job name
#SBATCH -N 4
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=4
#SBATCH -p compute_full_node


OPTIMIZER=lamb
PHASE=2


module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja
source activate pytorch


# Distributed system configuration
if [[ -z "${NODEFILE}" ]]; then
    if [[ -n "${SLURM_NODELIST}" ]]; then
        NODEFILE=/tmp/imagenet_slurm_nodelist
        scontrol show hostnames $SLURM_NODELIST > $NODEFILE
    elif [[ -n "${COBALT_NODEFILE}" ]]; then
        NODEFILE=$COBALT_NODEFILE
    fi
fi
if [[ -z "${NODEFILE}" ]]; then
    MAIN_RANK=$HOSTNAME
    RANKS=$HOSTNAME
    NNODES=1
else
    MAIN_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

if [[ -z "$PRUNE_INPUTS" ]]; then
    OUTPUT_DIR="results/phase${PHASE}_${OPTIMIZER}_${NNODES}nodes"
else
    OUTPUT_DIR="results/phase${PHASE}_${OPTIMIZER}_prune_inputs_${NNODES}nodes"
fi

CURRENT_DIR=$(pwd)

LOAD="module load anaconda3 cuda/11.4.4 gcc/10.3.0 ninja; export OMP_NUM_THREADS=8; source activate pytorch; cd $CURRENT_DIR; "
# SINGULARITY="singularity exec --nv --writable --bind $PWD:/home/mozaffar /scratch/mozaffar/torch.sif"





RANK=0
for NODE in $RANKS; do
    LAUNCHER="python -m torch.distributed.launch --nproc_per_node=4 --nnodes=$NNODES --node_rank=$RANK --master_addr=$MAIN_RANK --master_port=4321\
        run_pretraining.py \
        --input_dir ./phase${PHASE} \
        --output_dir ${OUTPUT_DIR} \
        --config_file config/bert_pretraining_phase${PHASE}_config.json \
        --weight_decay 0.01 \
        --num_steps_per_checkpoint 100 \
        --optimizer ${OPTIMIZER} \
        ${PRUNE_INPUTS} 
        "
    FULL_CMD="$SINGULARITY $LAUNCHER"
    if [[ $NODE == $MAIN_RANK ]]; then
        echo $FULL_CMD
	    eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
	      ssh $NODE "cd $CURRENT_DIR; $LOAD $FULL_CMD" &
    fi
    RANK=$((RANK + 1))
done


wait