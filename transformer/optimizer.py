import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupLinearLR(LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_steps,
                 total_steps,
                 min_proportion=0.0,
                 last_epoch=-1,
                 verbose=False):
        
        self.warmup_steps = warmup_steps
        self.max_steps = (total_steps - min_proportion * warmup_steps) / (1.0 - min_proportion)
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] * 0.1 / self.warmup_steps for group in self.optimizer.param_groups]
        
        if self.last_epoch > self.max_steps:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        if self.last_epoch < self.warmup_steps:
            return [group['initial_lr'] * self.last_epoch / self.warmup_steps for group in self.optimizer.param_groups]
        else:
            return [group['initial_lr'] * (self.max_steps - self.last_epoch) / (self.max_steps - self.warmup_steps) for
                    group in self.optimizer.param_groups]
    
    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [base_lr * (self.max_steps - self.last_epoch) / (self.max_steps - self.warmup_steps) for base_lr
                    in self.base_lrs]


def create_optimizer_and_scheduler(model,
                                   init_lr: float,
                                   warmup_steps: int,
                                   total_steps: int,
                                   min_proportion: float = 0.0,
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
    scheduler = WarmupLinearLR(optimizer, warmup_steps, total_steps, min_proportion=min_proportion)
    
    return optimizer, scheduler
