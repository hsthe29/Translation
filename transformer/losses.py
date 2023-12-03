import torch


class MaskedLabelSmoothingLoss(torch.nn.Module):
    def __init__(self, weight: torch.FloatTensor | None = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'none', label_smoothing: float = 0.0):
        super(MaskedLabelSmoothingLoss, self).__init__()
        
        self.loss_fn = torch.nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor):
        loss = self.loss_fn(input.permute(0, 2, 1), target)
        mask = mask.to(dtype=torch.float32)
        loss = loss * mask
        return loss.sum() / mask.sum()
    