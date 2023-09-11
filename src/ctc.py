from typing import Any, Tuple
import torch
import random
import numpy as np
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


reduction = 'mean'
blank     = 0

class CustomCTCLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, inp_len, tgt_len, targets):
        ctx.save_for_backward(inp)
        inp_len  = torch.as_tensor(inp_len, dtype=torch.long)
        tgt_len = torch.as_tensor(tgt_len, dtype=torch.long)
        dt = inp.dtype
        inp = inp.double()  # we need the accuracy as we are not in logspace
        targets = targets.long()
        cum_tgt_len = tgt_len.cumsum(0)
        losses = []

        for i in range(inp.size(1)):
            inp_length = inp_len[i].item()
            target_length = tgt_len[i].item()
            cum_target_length = cum_tgt_len[i].item()

            targets_prime = targets.new_full((2 * target_length + 1,), blank)
            if targets.dim() == 2:
                targets_prime[1::2] = targets[i, :target_length]
            else:
                targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]

            probs = inp[:inp_length, i].exp()

            alpha = inp.new_zeros((target_length * 2 + 1,))
            alpha[0] = probs[0, blank]
            alpha[1] = probs[0, targets_prime[1]]
            mask_third = (targets_prime[:-2] != targets_prime[2:])
            for t in range(1, inp_length):
                alpha_next = alpha.clone()
                alpha_next[1:] += alpha[:-1]
                alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
                alpha = probs[t, targets_prime] * alpha_next

            losses.append(-alpha[-2:].sum().log()[None])
        output = torch.cat(losses, 0)
        if reduction == 'mean':
            print( (output / tgt_len.to(dtype=output.dtype, device=output.device)).mean() )
        elif reduction == 'sum':
            print( output.sum() )
        output = output.to(dt)
        output_mean = (output / tgt_len.to(dtype=output.dtype, device=output.device)).mean()
        return output_mean
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        inp = ctx.saved_tensors
        grad_inp = None
        return grad_inp
