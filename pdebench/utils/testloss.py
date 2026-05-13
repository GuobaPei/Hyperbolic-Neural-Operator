import torch


class TestLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(TestLoss, self).__init__()

        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        # Avoid inf/nan relative loss when y_norm is zero
        self.eps = 1e-12

    def abs(self, x, y):
        num_examples = x.size()[0]

        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        # Skip samples with ||y||=0(otherwise relative loss divides by zero and can produce inf/nan or huge gradients)
        mask = y_norms > self.eps
        y_norms_safe = torch.clamp(y_norms, min=self.eps)
        rel = diff_norms / y_norms_safe
        if mask.any():
            rel = rel[mask]
        if self.reduction:
            if self.size_average:
                return torch.mean(rel)
            else:
                return torch.sum(rel)

        # reduction=False: return per-sample relative loss, with 0 for invalid samples.
        out = diff_norms / y_norms_safe
        if (~mask).any():
            out = out.clone()
            out[~mask] = 0.0
        return out

    def __call__(self, x, y):
        return self.rel(x, y)
