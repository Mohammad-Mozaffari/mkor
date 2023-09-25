# MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates

We have tested MKOR on BERT-Large-Uncased Pretraining and ResNet-50 and have achieved up to 2.57x speedup in comparison to state-of-the-art baselines.

## BERT-Large-Uncased Pretraining
We use the NVIDIA BERT-Large-Uncased implementation from [this](https://github.com/gpauloski/BERT-PyTorch/tree/master) repo, and have integrated MKOR as an external optimizer to it. For getting the desired results, please use the hyperparameters mentioned in the paper. If any hyperparameter isn't mentioned in the paper, please use the default value from the original repo.


Scripts for fine-tuning the checkpoints on SQuAD dataset are in `bert/scripts` and instructions for fine-tuning the checkpoints on the GLUE dataset are avaialbe in `NVIDIA BERT fine-tuning GLUE.md` inside the `bert` folder.

Currently, gathering the datasets for BERT Pretraining might not be feasible, since the datasets are not publicly available. For accessing the datasets, please reach out the authors of MKOR or [KAISA](https://arxiv.org/pdf/2107.01739.pdf).

## ResNet-50 Training

We use the exact implementation of [this](https://github.com/gpauloski/kfac-pytorch/tree/v0.3.1) repo, and have integrated MKOR as an external optimizer to it.
You can find the code in `resnet` folder. For getting the desired results, please use the hyperparameters mentioned in the paper. If any hyperparameter isn't mentioned in the paper, please use the default value from the original repo.

