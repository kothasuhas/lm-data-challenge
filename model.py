from transformers import GPTNeoForCausalLM, GPT2LMHeadModel, AutoTokenizer, AutoConfig

def get_model(cfg):
  if cfg.model_name == 'gpt2':
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_inner=cfg.n_inner,
        n_embd=cfg.n_embd,
      )
    model = GPT2LMHeadModel(config)
  elif cfg.model_name == 'gptNeo':
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    config = AutoConfig.from_pretrained(
        "EleutherAI/gpt-neo-1.3B",
        num_layers=cfg.n_layer,
        num_heads=cfg.n_head,
        hidden_size=cfg.n_inner,
      )
    model = GPTNeoForCausalLM(config)
  
  return model




