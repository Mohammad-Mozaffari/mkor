import torch



def prune_layer(params, sparsity_ratio):
    if sparsity_ratio > 1:
        sparsity_ratio /= 100
    mask = torch.zeros_like(params)
    num_weights = mask.numel()
    num_pruned = int(num_weights * sparsity_ratio)
    _, nonzero_indices = torch.topk(torch.abs(params).view(-1), num_weights - num_pruned)
    mask.view(-1)[nonzero_indices] = 2.0
    return mask


def block_diagonal_mask(shape, num_blocks):
    mask = torch.zeros(shape)
    block_size = shape[0] // num_blocks
    for i in range(num_blocks):
        mask[i * block_size:(i + 1) * block_size, i * block_size:(i + 1) * block_size] = 1.0
    mask[i * block_size:, i * block_size:] = 1.0
    return mask


