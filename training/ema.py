import torch

@torch.no_grad()
def ema_update(target_model, online_model, tau=0.996):
    """
    target = tau * target + (1 - tau) * online
    """
    for t_param, o_param in zip(
        target_model.parameters(),
        online_model.parameters()
    ):
        t_param.data.mul_(tau).add_(o_param.data, alpha=1 - tau)
