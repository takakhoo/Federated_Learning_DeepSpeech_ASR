from typing import Any, Tuple
import torch
import random
import numpy as np
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


reduction = 'mean'
BLANK     = 0

class CustomCTCLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, inp_len, tgt_len, targets):
        # inp now is NOT log space, softmax space !, shape (T,N,C)
        inp_len  = torch.as_tensor(inp_len, dtype=torch.long)
        tgt_len = torch.as_tensor(tgt_len, dtype=torch.long)
        dt = inp.dtype
        inp = inp.double()  # we need the accuracy as we are not in logspace
        targets = targets.long()
        cum_tgt_len = tgt_len.cumsum(0)
        losses = []
        alpha_global = []

        for i in range(inp.size(1)):
            inp_length = inp_len[i].item()
            target_length = tgt_len[i].item()
            cum_target_length = cum_tgt_len[i].item()

            targets_prime = targets.new_full((2 * target_length + 1,), BLANK)
            if targets.dim() == 2:
                targets_prime[1::2] = targets[i, :target_length]
            else:
                targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]

            # probs = inp[:inp_length, i].exp()
            probs = inp[:inp_length, i] # assume in softmax space

            # alpha = inp.new_zeros((target_length * 2 + 1,))
            alpha   = inp.new_zeros((inp_length, target_length*2+1))
            alpha[0,0] = probs[0, BLANK]
            alpha[0,1] = probs[0, targets_prime[1]]

            mask_third = (targets_prime[:-2] != targets_prime[2:])
            for t in range(1, inp_length):
                alpha[t]   = alpha[t-1].clone()
                alpha[t,1:] += alpha[t-1,:-1].clone()
                alpha[t,2:]  += torch.where(mask_third, alpha[t-1,:-2], alpha[t-1].new_zeros(1))
                alpha[t] = probs[t, targets_prime] * alpha[t]

                # alpha_next = alpha.clone()
                # alpha_next[1:] += alpha[:-1]
                # alpha_next[2:] += torch.where(mask_third, alpha[:-2], alpha.new_zeros(1))
                # alpha = probs[t, targets_prime] * alpha_next

            losses.append(-alpha[-1,-2:].sum().log()[None])
            alpha_global.append(alpha)
        output = torch.cat(losses, 0)
        if reduction == 'mean':
            print( (output / tgt_len.to(dtype=output.dtype, device=output.device)).mean() )
        elif reduction == 'sum':
            print( output.sum() )
        output = output.to(dt)
        output_mean = (output / tgt_len.to(dtype=output.dtype, device=output.device)).mean()

        ctx.save_for_backward(inp,inp_len, tgt_len, targets,torch.stack(alpha_global,dim=0), torch.tensor(losses))
        return output_mean
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        inps, inp_lens, tgt_lens, targets, alphas, losses = ctx.saved_tensors
        grad_inp = torch.zeros_like(inps)

        # compute beta

        cum_tgt_len = tgt_lens.cumsum(0)

        dt = inps.dtype
        targets = targets.long()

        # lp_to_l = lambda idx,tgt: 0 if idx%2==0 else tgt[idx//2] # label prime -> label

        for i in range(inps.size(1)): # loop through each example in dataset
            inp_length        = inp_lens[i].item()
            target_length     = tgt_lens[i].item()
            cum_target_length = cum_tgt_len[i].item()

            # ========================================================================================================= = 
            # to do: remove this target prime array, not needed.
            # use the lp_to_l later.
            targets_prime = targets.new_full((2 * target_length + 1,), BLANK)
            if targets.dim() == 2:
                targets_prime[1::2] = targets[i, :target_length]
            else:
                targets_prime[1::2] = targets[cum_target_length - target_length:cum_target_length]
            # ==========================================================================================================
            probs = inps[:inp_length, i]
            # ==========================================================================================================
            alpha = alphas[i]
            beta = inps.new_zeros((inp_length, target_length * 2 + 1))
            beta[-1, -1] = probs[-1, BLANK]
            grad_inp[-1, i, BLANK] = alpha[-1, -1]*beta[-1, -1] 

            if target_length > 0 :
                beta[-1, -2] = probs[-1,targets_prime[-2] ] # tricky
                grad_inp[-1,i,targets_prime[-2]] = alpha[-1, -2] * beta[-1, -2]

            for t in reversed(range(0, inp_length-1)):
                for s in reversed(range(0, 2*target_length+1)):
                    b1 = beta[t+1,s]
                    b2 = 0 if s >= 2 * target_length else beta[t+1,s+1]
                    b3 = 0 if s >= 2 * target_length -1 or targets_prime[s] == targets_prime[s+2] else beta[t+1,s+2]
                    beta[t,s] = (b1+b2+b3) * probs[t,targets_prime[s]]

                    alpha_beta = alpha[t,s] * beta[t,s]
                    grad_inp[t,i,targets_prime[s]] += alpha_beta # t,n,c

            # make sure beta is correct by computing loss using beta
            loss = -(beta[0,0] + beta[0,1]).log()
            assert loss == losses[0]

            #-a - b - c - d target prime
	        #-abc target
            # done beta
            # print(beta.shape) 

            # return -torch.div(grad_inp, torch.pow(inps,2)) * torch.exp(losses), None, None, None

            for t in range(inp_length):
                for c in range(inps.shape[-1]):
                    grad_inp[t,i,c] = -1/(inps[t,i,c]**2) *  grad_inp[t,i,c] * torch.exp(losses[i])
            return grad_inp,  None, None, None
