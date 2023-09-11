import torch
from torch import Tensor


class CTCLossGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths):
        ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths)
        return 3 * log_probs ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        log_probs, _, _, _ = ctx.saved_tensors
        return grad_output * 6 * log_probs, None, None, None, None


class CTCLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor, blank=0):
        # out = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        batch_size = targets.size(0)
        max_target_length = targets.size(1)
        max_input_length = log_probs.size(0)
        targets_prime = targets.new_full((batch_size, 2 * max_target_length + 1,), blank)
        targets_prime[:, 1::2] = targets[:, :max_target_length]
        probs = log_probs.exp()
        
        # Initialization
        alpha = log_probs.new_zeros((batch_size, max_target_length * 2 + 1, ))
        alpha[:, 0] = probs[0, :, blank]
        alpha[:, 1] = probs[0].gather(1, targets_prime[:, 1].unsqueeze(-1)).squeeze(-1)
        mask_third = targets_prime[:, :-2] != targets_prime[:, 2:]
        for t in range(1, max_input_length):
            alpha_next = alpha.clone()
            alpha_next[:, 1:] += alpha[:, :-1]
            alpha_next[:, 2:] += torch.where(mask_third, alpha[:, :-2], torch.zeros_like(alpha[:, :-2]))
            alpha = probs[t].gather(1, targets_prime) * alpha_next
        out = -alpha[:, -2:].sum(-1).log()
        out = (out / target_lengths).mean()
        ctx.save_for_backward(log_probs, targets, input_lengths, target_lengths, alpha)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        log_probs, targets, input_lengths, target_lengths, alpha = ctx.saved_tensors
        return grad_output * CTCLossGradFn.apply(
            log_probs, targets, input_lengths, target_lengths), \
            None, None, None
    

def compute_grad_loss(inputs, model, loss_fn, fl_grad):
    """
    Args:
        inputs
        model
        loss_fn
        fl_grad

    Returns grad_loss = ||grad_computed - grad_org||_2
    """
    log_probs = torch.nn.functional.log_softmax(
        model(inputs).permute(1, 0, 2))
    loss = loss_fn(log_probs)
    weights = [w for w in model.parameters() if w.requires_grad]
    weights_grad = torch.autograd.grad(loss, weights, create_graph=True)
    weights_grad = torch.cat([wg.view(-1) for wg in weights_grad], -1)
    grad_loss = torch.norm(weights_grad - fl_grad)
    return grad_loss


class TestCustomGradFn:
    def test_grad(self,
                  dim=8,
                  batch_size=2,
                  vocab_size=5,
                  input_max_len=10,
                  target_max_len=5):
        inputs = torch.autograd.Variable(
            torch.randn(batch_size, input_max_len, dim),
            requires_grad=True)  # B x L x d
        input_lengths = torch.ones(batch_size, dtype=torch.int) * input_max_len
        targets = torch.randint(1, vocab_size, (batch_size, target_max_len))
        target_lengths = torch.ones(batch_size, dtype=torch.int) * target_max_len
        
        model = torch.nn.Sequential(
            torch.nn.Linear(dim, 10),
            torch.nn.Linear(10, vocab_size))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        fl_grad = torch.zeros(num_params)

        def loss_fn(x):
            return CTCLossFn.apply(x, targets, input_lengths, target_lengths)

        # compute gradient with our custom implementation
        grad_loss = compute_grad_loss(inputs, model, loss_fn, fl_grad)
        grad_loss.backward(retain_graph=True)
        x_grad = inputs.grad

        # compute gradient with finite difference (numerical)
        inputs_1 = inputs
        inputs_2 = inputs + torch.randn_like(inputs) * 0.01
        grad_loss_1 = compute_grad_loss(inputs_1, model, loss_fn, fl_grad)
        grad_loss_2 = compute_grad_loss(inputs_2, model, loss_fn, fl_grad)
        x_grad_num = (grad_loss_1 - grad_loss_2) / (inputs_1 - inputs_2)
        print("snr", 10 * torch.log10(torch.mean(torch.abs(x_grad) / torch.norm(x_grad - x_grad_num))).detach().numpy())
        exit()

    def test_ctc(self,
                 batch_size=3,
                 vocab_size=5,
                 input_max_len=10,
                 target_max_len=5):
        inputs = torch.randn(input_max_len, batch_size, vocab_size)
        inputs = torch.nn.functional.log_softmax(inputs)
        input_lengths = torch.ones(batch_size, dtype=torch.int) * input_max_len
        targets = torch.randint(1, vocab_size, (batch_size, target_max_len))
        target_lengths = torch.ones(batch_size, dtype=torch.int) * target_max_len

        ctc_built_in = torch.nn.functional.ctc_loss(inputs, targets, input_lengths, target_lengths)
        ctc_custom = CTCLossFn.apply(inputs, targets, input_lengths, target_lengths)
        print(ctc_built_in, ctc_custom)
        exit()
        assert ctc_built_in - ctc_custom < 1e-5
 
    # def test_gradcheck(self):
    #     fn = CTCLossFn.apply
    #     inputs = torch.autograd.Variable(torch.randn(1, 10, 8, dtype=torch.float64), requires_grad=True)
    #     torch.autograd.gradcheck(fn, (inputs, None, None, None))
