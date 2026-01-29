import torch
import random

def mask_vision_patches(patch_embeddings, mask_ratio=0.6):
    """
    patch_embeddings: (B, N, D)
    Returns:
        visible_patches
        visible_indices
        masked_indices
    """
    B, N, D = patch_embeddings.shape
    num_mask = int(mask_ratio * N)

    visible_patches = []
    visible_indices = []
    masked_indices = []

    for b in range(B):
        idx = list(range(N))
        random.shuffle(idx)

        mask_idx = idx[:num_mask]
        vis_idx = idx[num_mask:]

        visible_patches.append(patch_embeddings[b, vis_idx])
        visible_indices.append(vis_idx)
        masked_indices.append(mask_idx)

    return visible_patches, visible_indices, masked_indices


def mask_text_spans(input_ids, attention_mask, mask_ratio=0.3, span_len=(3, 7)):
    """
    input_ids: (B, T)
    attention_mask: (B, T)
    Returns masked versions (tokens removed, not replaced)
    """
    B, T = input_ids.shape

    masked_inputs = []
    masked_attention = []
    masked_positions = []

    for b in range(B):
        valid_len = attention_mask[b].sum().item()
        num_to_mask = int(valid_len * mask_ratio)

        positions = []
        i = 1  # skip [CLS]
        while len(positions) < num_to_mask and i < valid_len - 1:
            span = random.randint(*span_len)
            span = min(span, valid_len - i - 1)
            positions.extend(range(i, i + span))
            i += span + random.randint(1, 3)

        positions = set(positions[:num_to_mask])
        keep = [i for i in range(T) if i not in positions]

        masked_inputs.append(input_ids[b, keep])
        masked_attention.append(attention_mask[b, keep])
        masked_positions.append(list(positions))

    return masked_inputs, masked_attention, masked_positions
