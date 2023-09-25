cd ../

OPTIMIZER=sgd
BATCH_SIZE=32

TRAIN_DIR=/scratch/m/mmehride/mozaffar/imagenet/train
VAL_DIR=/scratch/m/mmehride/mozaffar/imagenet/val

torchrun --nproc_per_node=4 torch_imagenet_resnet.py --train-dir $TRAIN_DIR --val-dir $VAL_DIR --optimizer $OPTIMIZER --batch-size $BATCH_SIZE --log-dir logs/local_run