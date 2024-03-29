from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('kothasuhas/QuRatedPajama_c4', split='train')

print('Saving the modified dataset to disk...')
dataset.save_to_disk('QuRatedPajama_c4_modified')