import os
import glob
import torch
import datasets
import pandas as pd
import numpy.random as npr
from functools import partial
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def do_tokenize(examples, tokenizer):
    """
    Tokenize pairs of sentences using a specified tokenizer.

    Args:
        examples (dict): A dictionary containing sentence pairs with keys "sentence_A" and "sentence_B".
        tokenizer: A tokenizer object with a callable interface for tokenizing text.

    Returns:
        dict: Tokenized outputs with padding, truncation, and a maximum length of 128 tokens.
    """
    return tokenizer(examples["sentence_A"], examples["sentence_B"],
                     padding="max_length", max_length=128, truncation=True)


def preprocess_single_dataset(input_path: str,
                              shuffle: bool,
                              test_split: float,
                              random_generator: npr.default_rng) -> dict:
    """
    Preprocess a dataset by reading, filtering, shuffling, and splitting into train, dev, and test sets.

    Args:
        input_path (str): Path to the input JSON file.
        shuffle (bool): Whether to shuffle sentence pairs.
        test_split (float): Proportion of the dataset to use for the test set.
        random_generator (npr.default_rng): Random number generator for reproducibility.

    Returns:
        dict: A dictionary containing the processed dataset splits as `datasets.Dataset` objects.
    """
    df_dict = dict()
    dataset_dict = datasets.DatasetDict()

    # read BLiMP files
    df = pd.read_json(input_path, lines=True)
    # filter and rename columns
    df = df.filter(
        items=["sentence_good", "sentence_bad", "UID"]
    )
    df = df.rename(
        columns={"sentence_good": "sentence_A", "sentence_bad": "sentence_B"}
    )
    # split dataset into train, dev, and test set
    df_dict["train"], df_dict["test"] = train_test_split(df, test_size=test_split)
    df_dict["train"], df_dict["dev"] = train_test_split(df_dict["train"],
                                                        test_size=len(df_dict["test"]))
    # prepare dataset
    for split, dataframe in df_dict.items():

        # reset df indices and keep old columns
        dataframe.reset_index(inplace=True, drop=False)

        # randomly generate binary labels to show which sentence will be grammatical
        labels = random_generator.integers(
            low=0, high=1, endpoint=True, size=len(dataframe)
        )

        # if shuffle, make sure that no two sentences end up next to their original pair
        if shuffle is True:
            # solution from https://stackoverflow.com/questions/27474921/compare-two-columns-using-pandas
            while True:
                shuffled_series = random_generator.permutation(dataframe["sentence_B"].values)
                retained_pairs = dataframe["sentence_B"][(dataframe["sentence_B"] == shuffled_series)]
                if len(retained_pairs):
                    continue
                else:
                    dataframe["sentence_B"] = shuffled_series
                    break

        # swap sentences according to the labels assigned
        for idx, value in enumerate(labels):
            if value == 1:
                dataframe.loc[idx, ["sentence_A", "sentence_B"]] = (
                    dataframe.loc[idx, ["sentence_B", "sentence_A"]].values)

        # add labels and convert to dataset
        dataframe["label"] = labels
        dataset = datasets.Dataset.from_pandas(dataframe)
        dataset_dict[split] = dataset

    return dataset_dict


def preprocess_whole_dataset(input_dir: str,
                             shuffle: bool,
                             random_generator: npr.default_rng,
                             seed: int = 42,
                             tokenize: bool = True,
                             tokenizer: AutoTokenizer = None,
                             test_split: float = 0.1) -> datasets.DatasetDict:
    """
    Preprocess and optionally tokenize an entire dataset directory.

    Args:
        input_dir (str): Directory containing the dataset files.
        shuffle (bool): Whether to shuffle sentence pairs within each dataset.
        random_generator (npr.default_rng): Random number generator for reproducibility.
        tokenizer (AutoTokenizer): Tokenizer to use if tokenization is enabled.
        seed (int, optional): Seed for shuffling the combined dataset. Default is 42.
        tokenize (bool, optional): Whether to tokenize the dataset. Default is True.
        test_split (float, optional): Proportion of each dataset to use for the test set. Default is 0.1.

    Returns:
        datasets.DatasetDict: A dictionary containing the processed and optionally tokenized dataset splits.
    """
    dataset_list = []
    overall_dataset = datasets.DatasetDict()

    # collect datasets
    for filepath in sorted(glob.glob(os.path.join(input_dir, "**"))):
        ds = preprocess_single_dataset(filepath, shuffle, test_split, random_generator)
        dataset_list.append(ds)

    # merge datasets
    for split in ["train", "dev", "test"]:
        overall_dataset[split] = datasets.concatenate_datasets(
            [d[split] for d in dataset_list]
        )
    overall_dataset = overall_dataset.shuffle(seed=seed)

    if tokenize is True:
        map_tokenize = partial(do_tokenize, tokenizer=tokenizer)
        overall_dataset = overall_dataset.map(map_tokenize, batched=True)
        overall_dataset.set_format("torch")

    return overall_dataset
