# -*- coding: utf-8 -*-

from datasets import Dataset, DatasetDict
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
import datasets
import random
# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

import tarfile
import os

def load_dataset(directory, num_files=34):
    """
    Loads the dataset from .tar files in the specified directory.

    :param directory: The directory containing .tar files.
    :param num_files: The number of .tar files to process.
    :return: A list of dictionaries, each containing the file path and text content.
    """
    dataset = []

    for i in range(27, num_files + 1):
        tar_file = os.path.join(directory, f"data_{i}.tar")

        if os.path.exists(tar_file):
            with tarfile.open(tar_file, "r") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".txt.clean"):
                        f = tar.extractfile(member)
                        if f:
                            text_content = f.read().decode("utf-8").strip()
                            dataset.append({"file_path": member.name, "text": text_content})
        else:
            print(f"File not found: {tar_file}")

    return dataset

# Usage example
directory = "/content/drive/My Drive/Sport_llm_data_cleaned/subset/"  # Replace with the path to your dataset

def create_dataset_dict(dataset, test_size=0.0005, seed=2404):
    """
    Splits the dataset into training and validation sets and creates a DatasetDict.

    :param dataset: The complete dataset loaded from the .tar files.
    :param test_size: The proportion of the dataset to include in the validation set.
    :param seed: The random seed for reproducible splits.
    :return: A DatasetDict with keys 'train' and 'val' for the training and validation sets.
    """
    random.seed(seed)
    random.shuffle(dataset)
    val_size = int(len(dataset) * test_size)
    val_set = dataset[:val_size]
    train_set = dataset[val_size:]

    train_dataset = Dataset.from_dict({"text": [item['text'] for item in train_set]})
    val_dataset = Dataset.from_dict({"text": [item['text'] for item in val_set]})

    return DatasetDict({"train": train_dataset, "val": val_dataset})

# Usage example
#directory = "/content/drive/My Drive/Sport_llm_data_cleaned/subset/"  # Replace with the path to your dataset
my_dataset = load_dataset(directory)
split_dataset = create_dataset_dict(my_dataset)

enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

save_dir = "/content/drive/My Drive/Sport_llm_data_cleaned/"  # Replace with the path to your dataset

# we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(save_dir), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 256

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

tokenized.items()
