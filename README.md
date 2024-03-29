# LLM Data Curation Challenge 

You will be given chunked C4 data that is annotated along many different axes

- Writing Style
- Facts and Trivia
- Educational Value
- Required Expertise
- [Cluster](https://huggingface.co/datasets/princeton-nlp/QuRatedPajama-1B_tokens_for_analysis/blob/main/cluster_checkpoint-1M_docs_for_analysis-k25/top_terms_with_title.csv)

Track 1, Ordering - Can you find an ordering of the data such that the final loss is lowest on a held-out evaluation set? Submit a function that orders the data for the best validation perplexity after 1-pass over the data.

Track 2, Filtering - Can you find a subset of the data to train on that's better than training on the entire dataset? Submit a subset of the data such that training on a random ordering of this data for a fixed number of steps (say 20k) gets optimal validation perplexity.

I need to give tons of credit to the awesome work of [QuRating](https://arxiv.org/abs/2402.09739) for annotating all of this data and Bingbin Liu for setting up a minimal training code infrastructure! This repo is a very lightweight wrapper around this effort.

## Setup

Create a conda environment (very vanilla huggingface, torch, etc so you might be able to skip this)

```
conda env create -f environment.yml
```

Modify the order or contents of the dataset by playing around in `modify_data.py` (possibly poking around using `inspect_data.py`) and then run the following. Keep filtering submissions in `filter_data.py` and ordering submissions in `order_data.py`.

```
python modify_data.py
```

Train a model determined by the settings in `conf/config.yaml`

```
python train.py
```

Evaluate a model determined by the settings in `conf/eval_config.yaml`

```
python eval.py
```