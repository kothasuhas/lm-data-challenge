from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('kothasuhas/QuRatedPajama_c4', split='train')

# example filtering where we only take points with good writing style and facts and trivia average

dataset = dataset.add_column('score', [0.0]*len(dataset))

def get_score(example):
    return example['writing_style_average'] + example['facts_and_trivia_average'] >= 0.0

dataset = dataset.filter(get_score)

print('Saving the modified dataset to disk...')
dataset.save_to_disk('QuRatedPajama_c4_filtered')