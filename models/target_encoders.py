import copy
import torch.nn as nn

def build_target_encoder(online_encoder):
    target = copy.deepcopy(online_encoder)
    for p in target.parameters():
        p.requires_grad = False
    return target
