from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('kothasuhas/QuRatedPajama_c4', split='train')

# example filtering where we sort by sum of writing style and facts and trivia average

dataset = dataset.add_column('score', [0.0]*len(dataset))

def get_score(example):
    example['score'] = example['writing_style_average'] + example['facts_and_trivia_average']
    return example

dataset = dataset.map(get_score)
dataset = dataset.sort('score')

print('Saving the modified dataset to disk...')
dataset.save_to_disk('QuRatedPajama_c4_modified')