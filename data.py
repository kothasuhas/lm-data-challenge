import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk


def get_loader(load_type, dataset_name, batch_size,
              split='train', shuffle=True):
  if load_type == 'huggingface':
    """
    Get a dataloader from a HuggingFace dataset
    """
    dataset = load_dataset(dataset_name)
    dataset = dataset[split]
  elif load_type == 'load_from_disk':
    dataset = load_from_disk(dataset_name)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return loader


def get_dataset(load_type, dataset_name,
              split='train',):
  if load_type == 'huggingface':
    """
    Get a dataloader from a HuggingFace dataset
    """
    dataset = load_dataset(dataset_name)
    dataset = dataset[split]
  elif load_type == 'load_from_disk':
    dataset = load_from_disk(dataset_name)
  return dataset

# class QuRateLoader:
#   def __init__(dataset, max_len, batch_size, shuffle):
#     self.dataset = dataset
#     self.max_len = max_len


def collate_fn(batch):
  # get texts
  input_ids = [item['input_ids'] for item in batch]
  # input_batch = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
  input_batch = torch.tensor(input_ids).long()
  shifted_input_batch = input_batch[:, :-1]
  shifted_label_batch = input_batch[:, 1:]
  # minus 1 because of the shift
  lengths = [item['length']-1 for item in batch]
  domains = [item['source_domain'] for item in batch]
  
  result = {'input': shifted_input_batch,
            'labels': shifted_label_batch,
            'lengths': lengths,
            'domains': domains}
  return result
  

