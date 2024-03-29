from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# dataset = load_dataset('kothasuhas/QuRatedPajama_c4', split='train')
dataset = load_from_disk('/data/suhas_kotha/lm-data-challenge/QuRatedPajama_c4_modified')

print(dataset[0])
