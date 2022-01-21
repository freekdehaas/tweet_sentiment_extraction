import pandas as pd
import numpy as np
import os

# Pytorch
from transformers import AlbertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

import opts

opt = opts.parse_opt()

train_path = os.path.join(opt.train_data, "train.csv")
train_df = pd.read_csv(train_path)
train_df.drop(
    314, inplace=True
)  # This row was found to have 'nan' values, so dropping it
train_df.reset_index(drop=True, inplace=True)
train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"]

test_path = os.path.join(opt.test_data, "test.csv")
test_df = pd.read_csv(test_path).reset_index(drop=True)
test_df["text"] = test_df["text"].astype(str)


class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        # data loading
        self.df = df
        self.selected_text = "selected_text" in df
        self.tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        self.max_length = opt.max_length

    def __len__(self):
        # len(dataset) i.e., the total number of samples
        return len(self.df)

    def get_data(self, row):
        # processing the data
        text = " " + " ".join(row.text.lower().split())  # clean the text
        encoded_input = self.tokenizer.encode(text)  # the sentence to be encoded

        sentiment_id = {
            "positive": 1313,
            "negative": 2430,
            "neutral": 7974,
        }  # stating the ids of the sentiment values

        # print ([list((i, encoded_input[i])) for i in range(len(encoded_input))])
        """
        # The input_ids are the sentence or sentences represented as tokens. 
        # There are a few BERT special tokens that one needs to take note of:

        # [CLS] - Classifier token, value: [101] 
        # [SEP] - Separator token, value: [102]
        # [PAD] - Padding token, value: 0

        # Bert expects every row in the input_ids to have the special tokens included as follows:

        # For one sentence as input:
        # [CLS] ...word tokens... [SEP]

        # For two sentences as input:
        # [CLS] ...sentence1 tokens... [SEP]..sentence2 tokens... [SEP]
        """

        input_ids = (
            [101] + [sentiment_id[row.sentiment]] + [102] + encoded_input.ids + [102]
        )

        """
        id: unique identifier for each token
        offset: starting and ending point in a sentence
        """

        # ID offsets
        offsets = (
            [(0, 0)] * 3 + encoded_input.offsets + [(0, 0)]
        )  # since first 3 are [CLS] ...sentiment tokens... [SEP]

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [0] * pad_len
            offsets += [(0, 0)] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        masks = torch.where(input_ids != 0, torch.tensor(1), torch.tensor(0))
        """
        # The attention mask has the same length as the input_ids(or token_type_ids). 
        # It tells the model which tokens in the input_ids are words and which are padding. 
        # 1 indicates a word (or special token) and 0 indicates padding.

        # For example:
        # Tokens: [101, 7592, 2045, 1012,  102,    0,    0,    0,    0,    0]
        # Attention mask: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        """

        masks = torch.tensor(masks, dtype=torch.long)
        offsets = torch.tensor(offsets, dtype=torch.long)

        return input_ids, masks, text, offsets

    def get_target_ids(self, row, text, offsets):
        # preparing data only for the training
        selected_text = " " + " ".join(row.selected_text.lower().split())

        string_len = len(selected_text) - 1

        idx0 = None
        idx1 = None

        for ind in (
            position for position, line in enumerate(text) if line == selected_text[1]
        ):
            if " " + text[ind : ind + string_len] == selected_text:
                idx0 = ind
                idx1 = ind + string_len - 1
                break

        char_targets = [0] * len(text)
        if idx0 != None and idx1 != None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        # Start and end tokens
        target_idx = []
        for k, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                try:
                    target_idx.append(k)
                except:
                    continue

        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        return selected_text, targets_start, targets_end

    def __getitem__(self, index):  # addressing each row by its index
        # dataset[index] i.e., generates one sample of data
        data = {}
        row = self.df.iloc[index]

        ids, masks, text, offsets = self.get_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["text"] = text
        data["offsets"] = offsets
        data["sentiment"] = row.sentiment

        if self.selected_text:  # checking if selected text exists
            # This part only exists in the training
            selected_text, start_index, end_index = self.get_target_ids(
                row, text, offsets
            )
            data["start_index"] = start_index
            data["end_index"] = end_index
            data["selected_text"] = selected_text

        return data


def train_val_dataloaders(df, train_idx, val_idx, batch_size, tokenizer):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TextDataset(train_df, tokenizer, opt.max_lenght),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # to avoid multi-process, keep it at 0
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        TextDataset(val_df, tokenizer, opt.max_lenght),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict


def test_loader(df, tokenizer, batch_size=opt.test_batch_size):
    loader = torch.utils.data.DataLoader(
        TextDataset(test_df, tokenizer, opt.max_lenght),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return loader
