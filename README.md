# English-Vietnamese Bilingual Translation with Positional Separated Attention Transformer 
Continued development from the repo [NeuralMachineTranslation](https://github.com/hsthe29/NeuralMachineTranslation)

```
@github{Translation,
  author    = {The Ho Sy},
  title     = {English-Vietnamese Bilingual Translation with Transformer},
  year      = {2023},
  url       = {https://github.com/hsthe29/Translation},
}
```

# Model Architecture
- Modified from Vanilla Transformer's Architecture

# Data
- See [PhoMT](https://github.com/VinAIResearch/PhoMT).

# Training Task:
- Target Masked Translation Modeling (Target MTM)
  - Target MTM:
    ```
    Training: 
      input: ["en<s>", "How", "are", "you?", "</s>"]
      target in: ["vi<s>", "Bạn", "có", "<mask>", "không?", "</s>"]
      target out: ["Bạn", "có", "khỏe", "không?", "</s>", "<pad>"]
    Inference:
      input: ["en<s>", "How", "are", "you?", "</s>"]
      target in: ["vi<s>"]
      Autoregressive -> full target out: ["vi<s>", Bạn", "có", "khỏe", "không?", "</s>"]
    ```

# Bilingual Vocabulary:
- English sentence start token: `en<s>`
- Vietnamese sentence start token: `vi<s>`
- End sentence token: `</s>`
- Mask token: `<mask>` for task MLM (training only)

# Example:
- Natural english `"Hello, how are you?"`, target start token `"vi<s>"`: 
  - Transform to `"en<s> Hello, how are you? </s>"`
  - Target: `"vi<s> Xin chào, bạn có khỏe không? </s>""`
- Natural vietnamese `"Xin chào, bạn có khỏe không?"`, target start token `"en<s>"`: 
  - Transform to `"vi<s> Xin chào, bạn có khỏe không? </s>"`
  - Target: `"en<s> Hello, how are you? </s>""`

# Model configuration
- See file [config.py](transformer/model/config.py) and [configV1.json](assets/config/configV1.json)

# Preload dataset
- Because of the large amount of data, my resources are limited, so I have to process and segment the data to be able to train the model.
- Preload parameter:
  - seed: a seed to create randoms from random generator 
  - shuffle: if True, the dataset will be shuffled before chunked
  - chunk_size: size of each chunk

# Training parameters
- Epochs = 10
- Batch size = 8
- Accumulation gradient steps = 4
- AdamW optimizer with WarmupLinearScheduler
- Max learning rate = 2e-4
- Training arguments:
  - config 
  - load_prestates 
  - epochs 
  - init_lr
  - train_data_dir
  - val_data_dir
  - train_batch_size
  - val_batch_size
  - print_steps
  - validation_steps
  - max_warmup_steps
  - gradient_accumulation_steps
  - save_state_steps
  - weight_decay
  - warmup_proportion
  - min_proportion
  - use_gpu
  - max_grad_norm
  - save_ckpt
  - ckpt_loss_path
  - ckpt_bleu_path
  - state_path

# Web server
- Using [Flask](https://github.com/pallets/flask) to deploy a simple web server run on `localhost` that provides bilingual translation and visualizes attention weights between pairs of sentences
- Run: `$ python run_app.py` or `$ python3 run_app.py`
