import torch
from torch.optim.lr_scheduler import LambdaLR


class WarmupLinearLR(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `total_steps - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupLinearLR, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.total_steps - step) / float(max(1.0, self.total_steps - self.warmup_steps)))
    

def create_optimizer_and_scheduler(model,
                                   init_lr: float,
                                   warmup_steps: int,
                                   total_steps: int,
                                   weight_decay: float = 0.001):
    """Returns: optimizer, scheduler"""
    no_decay = ['bias', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=init_lr, weight_decay=0.0)
    scheduler = WarmupLinearLR(optimizer, warmup_steps, total_steps)
    
    return {"optimizer": optimizer, "scheduler": scheduler}
