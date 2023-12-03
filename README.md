# English-Vietnamese Bilingual Translation with Positional Separated Attention Transformer 
```
@github{Translation,
  author    = {The Ho Sy},
  title     = {English-Vietnamese Bilingual Translation with Transformer},
  year      = {2023},
  url       = {https://github.com/hsthe29/Translation},
}
```

# Data
- [PhoMT]()

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

# Training parameters
- Epochs = 10
- Batch size = 8
- Accumulation gradient steps = 4
- AdamW optimizer with WarmupLinearScheduler
- Max learning rate = 2e-4