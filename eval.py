# Training loop for language models
# using hydra for configuration management

import os
import hydra
from omegaconf import DictConfig, OmegaConf
# get transformers from huggingface
from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM, GPT2LMHeadModel, GPTNeoForCausalLM
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from math import ceil
from tqdm import tqdm
from datetime import datetime

import pdb

# local imports
# the entry dir should be `real_data_training`
from data import get_loader, get_dataset, collate_fn
from model import get_model, get_model_from_pretrained

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# training with the next word prediction task
def eval(model, data_loader, n_steps, criterion, device, cfg):
    model.eval()
    total_loss = 0
    global_step_cnt = 0

    # first_batch = 1
    for batch in tqdm(data_loader, total=n_steps):
        global_step_cnt += 1

        inputs, labels = batch['input'], batch['labels']
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        logits = outputs.logits

        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()
        
        if global_step_cnt >= n_steps:
            break

    print (total_loss / global_step_cnt)



if __name__ == "__main__":
    @hydra.main(config_path="conf", config_name="eval_config")
    def main(cfg):
        # set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # set seeds
        set_seeds(cfg.seed)

        # get data loader
        # data_loader = get_loader(cfg.load_type, cfg.dataset_name, cfg.batch_size)
        dataset = get_dataset(cfg.load_type, cfg.dataset_name)
        data_loader = DataLoader(dataset,
            batch_size=cfg.batch_size, shuffle=cfg.shuffle,
            collate_fn=collate_fn,
            num_workers=8)

        # get model
        model = get_model_from_pretrained(cfg)
        # check the number of parameters in the model
        num_params = sum(p.numel() for p in model.parameters())
        if num_params > 1e9:
          print(f"# params: {num_params/1e9}B")
        else:
          print(f"# params: {num_params/1e6}M")
        model = model.to(device)

        # set loss function
        criterion = nn.CrossEntropyLoss()

        # eval
        eval(model, data_loader, cfg.n_steps, criterion, device, cfg)

    main()