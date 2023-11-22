from typing import List, Tuple

import torch


def collate_tensor_fn_accumulate(batch):
    """Does the same as default collate for tensor of torch, except that instead of stacking on an new dimension,
    stacks on the first dimension.
    """
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem.storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch) * elem.size(0), *list(elem.size()[1:]))
    return torch.cat(batch, 0, out=out)


def collate_fn_gray_scale(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
    """
    Collate function for dataloader that also saves the values for each batch element. Collate fn linked to
    the gray scale datasets.

    Args:
        batch (list): input batch

    Returns:
        list:
            default_collate output, but for each element we added the values if it exists in the batch.
    """
    output = tuple(collate_tensor_fn_accumulate(samples) for samples in zip(*batch))

    all_values = []
    cur_idx = 0
    indexes = [0]

    original_inputs = []
    original_targets = []

    for idx in range(len(batch)):
        input_tensor = batch[idx][0]
        target_tensor = batch[idx][1]

        all_values.append(input_tensor.gray_values)

        cur_idx += len(input_tensor.gray_values)
        indexes.append(cur_idx)

        original_inputs.append(input_tensor.original)
        original_targets.append(target_tensor.original)

    all_values = torch.cat(all_values)
    output[0].gray_values = all_values
    output[0].indexes = indexes
    output[0].original = torch.stack(original_inputs)
    output[1].original = torch.stack(original_targets)

    return output
