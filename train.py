import math

import torch.cuda
from torch.utils.data import DataLoader

from argument import CommandLineArgument, CommandLineFlag
from transformer import TransformerOutput
from transformer.data import load_dataset, DataCollator
from transformer.losses import MaskedLabelSmoothingLoss
from transformer.metrics import MaskedAccuracyScore, BLEUScore, Metric
from transformer.model import BilingualTokenizer, TransformerConfig, Transformer
from transformer.optimizer import create_optimizer_and_scheduler

argument_parser = CommandLineArgument()
argument_parser.define("config", "assets/config/configV1.json", str)

argument_parser.define("load_prestates", False, bool)

argument_parser.define("epochs", 20, int)
argument_parser.define("init_lr", 1e-4, float)

argument_parser.define("train_data_dir", "./data/preload/PhoMT/train/", str)
argument_parser.define("val_data_dir", "./data/preload/PhoMT/dev/", str)

argument_parser.define("train_batch_size", 16, int)
argument_parser.define("val_batch_size", 32, int)
argument_parser.define("print_steps", 400, int)
argument_parser.define("validation_steps", 2000, int)
argument_parser.define("max_warmup_steps", 20_000, int)
argument_parser.define("gradient_accumulation_steps", 4, int)
argument_parser.define("save_state_steps", 2000, int)

argument_parser.define("weight_decay", 0.001, float)
argument_parser.define("warmup_proportion", 0.1, float)
argument_parser.define("use_gpu", True, bool)

argument_parser.define("max_grad_norm", 1.0, float)

argument_parser.define("save_ckpt", True, int)
argument_parser.define("ckpt_loss_path", "checkpoint/loss/model_state_dict.pt", str)
argument_parser.define("ckpt_bleu_path", "checkpoint/bleu/model_state_dict.pt", str)
argument_parser.define("state_path", "state/training_state_dict.pt", str)


def _train_step(model, batch, criterion: MaskedLabelSmoothingLoss, metrics: list[Metric]):
    input, target, label = batch
    
    assert label.mask.dtype == torch.int64
    
    outputs: TransformerOutput = model(input, target)
    
    logits = outputs.logits
    
    loss = criterion(logits, label.ids, label.mask)
    for _, metric in enumerate(metrics):
        metric.cumulate(logits, label.ids, label.mask)
    
    return loss


@torch.no_grad()
def _eval_step(model, batch, criterion: MaskedLabelSmoothingLoss, metrics: list[Metric]):
    input, target, label = batch
    
    assert label.mask.dtype == torch.int64
    
    outputs: TransformerOutput = model(input, target)
    
    logits = outputs.logits
    
    loss = criterion(logits, label.ids, label.mask)
    for _, metric in enumerate(metrics):
        metric.cumulate(logits, label.ids, label.mask)
    
    return loss


@torch.no_grad()
def do_evaluate(model, dataloader, device, tokenizer, config: TransformerConfig):
    model.eval()
    masked_accuracy_score = MaskedAccuracyScore()
    bleu_score = BLEUScore(tokenizer, max_n=4, weights=[0.25, 0.3, 0.25, 0.2])
    total_loss = 0.0
    criterion = MaskedLabelSmoothingLoss(label_smoothing=config.label_smoothing)
    validation_steps = len(dataloader)
    for step, batch in enumerate(dataloader):
        print(f"\r    - Validation step: {step + 1:>4d}/{validation_steps}", end="")
        loss = _eval_step(model,
                          batch.to(device),
                          criterion=criterion,
                          metrics=[masked_accuracy_score, bleu_score])
        total_loss += loss.item()
    
    loss = round(total_loss / validation_steps, 4)
    accuracy = round(masked_accuracy_score.get_score(), 4)
    bleu = round(bleu_score.get_score(), 4)
    
    return {"loss": loss,
            "accuracy": accuracy,
            "bleu": bleu}


def do_train():
    args: CommandLineFlag = argument_parser.parse()
    print("Training arguments:", args)
    
    config_file = getattr(args, "config", None)
    
    if config_file is None:
        raise ValueError("config has not been specified!")
    
    config = TransformerConfig.load(config_file)
    tokenizer = BilingualTokenizer(config)
    
    train_data_dir = getattr(args, "train_data_dir", None)
    val_data_dir = getattr(args, "val_data_dir", None)
    
    if train_data_dir is None:
        raise ValueError("train_data_dir has not been specified")
    
    train_ds = load_dataset(train_data_dir, config)
    val_ds = None if val_data_dir is None else load_dataset(val_data_dir, config)
    
    train_collator = DataCollator(tokenizer, config.mlm_probability)
    val_collator = DataCollator(tokenizer, None)
    
    train_batch_size = getattr(args, "train_batch_size", 1)
    val_batch_size = getattr(args, "val_batch_size", 1)
    
    train_dl = DataLoader(train_ds, collate_fn=train_collator, batch_size=train_batch_size, shuffle=False)
    
    val_dl = (DataLoader(val_ds, collate_fn=val_collator, batch_size=val_batch_size, shuffle=False)
              if val_ds is not None else None)
    
    epochs = getattr(args, "epochs")
    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
    warmup_proportion = getattr(args, "warmup_proportion", 0.1)
    steps_per_epoch = len(train_dl)
    total_steps = epochs * math.ceil(steps_per_epoch / gradient_accumulation_steps)
    max_warmup_steps = getattr(args, "max_warmup_steps")
    warmup_steps = min(max_warmup_steps, int(warmup_proportion * total_steps))
    
    use_gpu = getattr(args, "use_gpu", False)
    
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    
    print("Found device:", device)
    
    model = Transformer(config).to(device)
    
    optimizer_and_scheduler = create_optimizer_and_scheduler(model,
                                                             init_lr=getattr(args, "init_lr"),
                                                             warmup_steps=warmup_steps,
                                                             total_steps=total_steps,
                                                             weight_decay=getattr(args, "weight_decay"))
    
    optimizer = optimizer_and_scheduler["optimizer"]
    scheduler = optimizer_and_scheduler["scheduler"]
    
    monitor_loss = float("inf")
    monitor_bleu = 0.0
    
    criterion = MaskedLabelSmoothingLoss(label_smoothing=config.label_smoothing)
    masked_accuracy_score = MaskedAccuracyScore()
    
    global_steps = 0
    
    max_grad_norm = getattr(args, "max_grad_norm")
    print_steps = getattr(args, "print_steps", 1)
    validation_steps = getattr(args, "validation_steps", 1)
    save_state_steps = getattr(args, "save_state_steps", 1)
    save_ckpt = getattr(args, "save_ckpt", False)
    ckpt_loss_path = getattr(args, "ckpt_loss_path")
    ckpt_bleu_path = getattr(args, "ckpt_bleu_path")
    state_path = getattr(args, "state_path")
    
    history = {"loss": [],
               "acc": [],
               "val_loss": [],
               "val_acc": [],
               "val_bleu": []}
    
    left_off_epoch = -1
    left_off_step = -1
    found_left_off = True
    
    training_states = None
    
    load_prestates = getattr(args, "load_prestates", False)
    
    if load_prestates:
        found_left_off = False
        training_states = torch.load(state_path)
        model.load_state_dict(training_states["model"])
        optimizer.load_state_dict(training_states["optimizer"])
        scheduler.load_state_dict(training_states["scheduler"])
        
        monitor_loss = training_states["monitor_loss"]
        monitor_bleu = training_states["monitor_bleu"]
        history = training_states["history"]
        
        left_off_epoch = training_states["epoch"]
        left_off_step = training_states["step"]
        global_steps = training_states["global_steps"]
    
    print("****************Training Arguments ****************")
    print("* Epochs:", epochs)
    print("* Learning rate:", getattr(args, "init_lr"))
    print("* Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("* Warmup steps:", warmup_steps)
    print("* Total steps:", total_steps)
    print("* Use gpu:", use_gpu)
    print("* Continue training:", load_prestates)
    print("***************************************************")
    print()
    
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        if not found_left_off:
            if epoch < left_off_epoch:
                continue
        
        print(f"Epoch \033[92m{epoch + 1:>3d}/{epochs}\033[00m:")
        accumulation_steps = min(gradient_accumulation_steps, total_steps)
        epoch_update_steps = 0
        epoch_loss = 0.0
        for step, batch in enumerate(train_dl):
            if not found_left_off:
                if step == left_off_step:
                    found_left_off = True
                    epoch_update_steps = training_states["epoch_update_steps"]
                    epoch_loss = training_states["epoch_loss"]
                continue
            
            print(f"\r    - Step \033[96m{step + 1:>5d}/{steps_per_epoch}\033[00m:", end="")
            model.train()
            loss = _train_step(model, batch.to(device), criterion, [masked_accuracy_score])
            epoch_loss += loss.item()
            loss = loss / accumulation_steps
            loss.backward()
            
            if ((step + 1) % gradient_accumulation_steps == 0) or ((step + 1) == steps_per_epoch):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                accumulation_steps = min(gradient_accumulation_steps, steps_per_epoch - step - 1)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_update_steps += 1
                global_steps += 1
                
                if (epoch_update_steps % print_steps == 0) or (step + 1 == steps_per_epoch):
                    train_loss_value = round(epoch_loss / (step + 1), 4)
                    train_accuracy_value = round(masked_accuracy_score.get_score(), 4)
                    train_log_line = (f"\033[95m{'loss':>8s}\033[00m: {train_loss_value:<10.4f} "
                                      f"\033[95m{'accuracy':>12s}\033[00m: {train_accuracy_value:<10.4f}")
                    print()
                    print("        ", train_log_line)
                    
                    history["loss"].append((global_steps, train_loss_value))
                    history["acc"].append((global_steps, train_accuracy_value))
                
                if val_dl is not None and (
                        (epoch_update_steps % validation_steps == 0) or (step + 1 == steps_per_epoch)):
                    validation_outputs = do_evaluate(model, val_dl, device, tokenizer, config)
                    
                    val_loss = validation_outputs['loss']
                    val_accuracy = validation_outputs['accuracy']
                    val_bleu = validation_outputs['bleu']
                    
                    val_log_line = (f"\033[95m{'val_loss':>8s}\033[00m: {val_loss:<10.4f} "
                                    f"\033[95m{'val_accuracy':>12s}\033[00m: {val_accuracy:<10.4f} "
                                    f"\033[95m{'val_bleu':>8s}\033[00m: {val_bleu:<10.4f} ")
                    
                    print("\r", end="")
                    print("        ", val_log_line)
                    
                    history["val_loss"].append((global_steps, val_loss))
                    history["val_acc"].append((global_steps, val_accuracy))
                    history["val_bleu"].append((global_steps, val_bleu))
                    
                    if save_ckpt:
                        if val_loss < monitor_loss:
                            print(f"    # Update validation loss from {monitor_loss} to {val_loss}")
                            monitor_loss = val_loss
                            torch.save(model.state_dict(), ckpt_loss_path)
                        
                        if val_bleu > monitor_bleu:
                            print(f"    # Update bleu score from {monitor_bleu} to {val_bleu}")
                            monitor_bleu = val_bleu
                            torch.save(model.state_dict(), ckpt_bleu_path)
                
                if global_steps % save_state_steps == 0:
                    training_states = {
                        "max_epochs": epochs,
                        "epoch": epoch,
                        "step": step,
                        "epoch_update_steps": epoch_update_steps,
                        "global_steps": global_steps,
                        "epoch_loss": epoch_loss,
                        "monitor_loss": monitor_loss,
                        "monitor_bleu": monitor_bleu,
                        "history": history,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()}
                    
                    torch.save(training_states, state_path)
                    
                    print(f"# Save states to {state_path}")
            
            if (step + 1) == steps_per_epoch:
                print("# Epoch finish\n")
        
        masked_accuracy_score.reset()
    
    torch.save(history, "checkpoint/history.pt")


if __name__ == "__main__":
    do_train()
