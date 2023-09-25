cd ../

PHASE=2

export OMP_NUM_THREADS=8

python -m torch.distributed.launch --nproc_per_node=1 run_pretraining.py \
    --input_dir ./phase${PHASE} \
    --output_dir results/phase${PHASE}_local \
    --config_file config/bert_pretraining_phase${PHASE}_config.json \
    --weight_decay 0.01 \
    --num_steps_per_checkpoint 200 \
    --global_batch_size 512 \
    --local_batch_size 4