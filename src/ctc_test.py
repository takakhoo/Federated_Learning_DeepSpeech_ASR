import torch
from .ctc import CustomCTCLoss


class TestCTC:
    def test_ctc_val(self):
        device    = 'cpu'
        BS = 1
        T  = 50
        C_SIZE = 15

        tgt_len = [30]
        inp_len = [50]
        targets = torch.randint(1, C_SIZE , (sum(tgt_len),), dtype=torch.int)
        log_probs = torch.nn.Parameter((torch.randn(T, BS, C_SIZE, dtype=torch.float, device=device).log_softmax(2)))

        ctc_loss = torch.nn.CTCLoss()
        ctc_loss_custom = CustomCTCLoss.apply
        out_built_in = ctc_loss(log_probs, targets, inp_len, tgt_len)
        out_custom = ctc_loss_custom(log_probs, inp_len, tgt_len, targets)

        assert torch.norm(out_built_in - out_custom) < 1e-4

    def test_ctc_grad(self):
        ctc_loss_custom = CustomCTCLoss.apply
        device    = 'cpu'
        BS = 1
        T  = 50
        C_SIZE = 15

        tgt_len = [30]
        inp_len = [50]
        targets = torch.randint(1, C_SIZE , (sum(tgt_len),), dtype=torch.int)
        log_probs = torch.nn.Parameter((torch.randn(T, BS, C_SIZE, dtype=torch.float, device=device).log_softmax(2)))
        torch.autograd.gradcheck(ctc_loss_custom, (log_probs, inp_len, tgt_len, targets))
