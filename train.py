# Training loop for language models
# using hydra for configuration management

import os
import hydra
from omegaconf import DictConfig, OmegaConf
# get transformers from huggingface
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from math import ceil
from tqdm import tqdm
from datetime import datetime
import wandb 

import pdb

# local imports
# the entry dir should be `real_data_training`
from data import get_loader, get_dataset, collate_fn
from model import get_model

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# training with the next word prediction task
def train(model, data_loader, n_steps, optimizer, scheduler, criterion, macro_batch_size, device):
    model.train()
    total_loss = 0
    n_epochs = ceil(n_steps / len(data_loader))
    global_step_cnt = 0
    
    optimizer.zero_grad()

    
    for epoch in tqdm(range(n_epochs)):
      # first_batch = 1
      for batch in tqdm(data_loader):
          global_step_cnt += 1

          inputs, labels = batch['input'], batch['labels']
          inputs = inputs.to(device)
          labels = labels.to(device)

          outputs = model(inputs)
          logits = outputs.logits

          loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
          
          # take a gradient for the macrobatch
          loss.backward()
          if global_step_cnt % macro_batch_size == 0:
              optimizer.step()
              scheduler.step()
              optimizer.zero_grad()

          if global_step_cnt % 100 == 0 or global_step_cnt <= 3:
              model.eval()
              with torch.no_grad():
                logits_at_50 = logits[:, 50]
                labels_at_50 = labels[:, 50]
                loss_at_50 = criterion(logits_at_50, labels_at_50).item()
                logits_at_500 = logits[:, 500]
                labels_at_500 = labels[:, 500]
                loss_at_500 = criterion(logits_at_500, labels_at_500).item()
                losses = {
                    'loss': loss.item(),
                    'loss_at_50': loss_at_50, 'loss_at_500': loss_at_500,}
                domains = set(batch['domains'])
                # check per-domain loss
                for domain in domains:
                    domain_mask = torch.tensor([d == domain for d in batch['domains']]).to(device)
                    domain_logits = logits[domain_mask]
                    domain_labels = labels[domain_mask]
                    domain_loss = criterion(domain_logits.view(-1, domain_logits.size(-1)), domain_labels.view(-1)).item()
                    logits_at_50 = domain_logits[:, 50]
                    labels_at_50 = domain_labels[:, 50]
                    domain_loss_at_50 = criterion(logits_at_50, labels_at_50).item()
                    logits_at_500 = domain_logits[:, 500]
                    labels_at_500 = domain_labels[:, 500]
                    domain_loss_at_500 = criterion(logits_at_500, labels_at_500).item()
                    losses['loss_'+domain] = domain_loss
                    losses['loss_at_50_'+domain] = domain_loss_at_50
                    losses['loss_at_500_'+domain] = domain_loss_at_500
                wandb.log(losses, step=global_step_cnt)
              
              model.train()

          
          if global_step_cnt >= n_steps:
              break

    # return total_loss / len(data_loader)



if __name__ == "__main__":
    @hydra.main(config_path="conf", config_name="config")
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
        model = get_model(cfg)
        # check the number of parameters in the model
        num_params = sum(p.numel() for p in model.parameters())
        if num_params > 1e9:
          print(f"# params: {num_params/1e9}B")
        else:
          print(f"# params: {num_params/1e6}M")
        model = model.to(device)

        # set optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=cfg.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, cfg.n_steps)

        # set loss function
        criterion = nn.CrossEntropyLoss()

        # set wandb
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_name = cfg.wandb_dataname + '_' + timestamp
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
        pid = os.getpid()
        wandb_cfg['pid'] = pid 
        wandb.init(project='curriculum',
                   entity=cfg.wandb_entity,
                   config=wandb_cfg,
                   name=wandb_name)

        # train
        train(model, data_loader, cfg.n_steps, optimizer, scheduler, criterion, cfg.macro_batch_size, device)
        # print(f"Training loss: {loss}")

    main()