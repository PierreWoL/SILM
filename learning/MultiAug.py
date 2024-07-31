# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import os
import random
from logging import getLogger

import torch
from transformers import AutoTokenizer

from Utils import childList, subjectCol

logger = getLogger()
from torch.utils.data import Dataset
import pandas as pd
from learning.augment import augment

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}


class MultiCropTableDataset(Dataset):
    def __init__(self,
                 path,
                 nmb_crops: list,
                 percentage_crops: list,
                 size_dataset=-1,
                 shuffle_rate=0.2,
                 lm='roberta',
                 subject_column=False,
                 augmentation_methods="sample_cells_TFIDF",
                 column=False,
                 header=False,
                 return_index=False,
                 max_length=512):

        self.path = path
        self.column = column
        self.subject_column = subject_column
        self.header = header
        self.nmb_crops = nmb_crops
        self.percentage_crops = percentage_crops
        self.augmentation_methods = []

        total_number = 0
        for i in nmb_crops:
            total_number += i
        """
        Transfer augmentation methods list
        """

        if isinstance(augmentation_methods, str):

            self.augmentation_methods = [augmentation_methods] * total_number
        else:
            for index, num in enumerate(nmb_crops):
                self.augmentation_methods.extend([augmentation_methods[index]*num])



        """
        Create args for the augmentation methods
        """
        self.trans = self._create_transforms()

        """Initialize the data items"""
        self.samples = []
        table_name = [fn for fn in os.listdir(path) if '.csv' in fn]
        if column is True:
            for fn in table_name:
                fn_table = os.path.join(self.path, fn)
                columns = pd.read_csv(fn_table).columns
                for col in columns:
                    self.samples.append(f"{fn}|{col}")
        else:
            self.samples = [fn for fn in os.listdir(path) if '.csv' in fn]
        if size_dataset >= 0:
            self.samples = childList(self.samples, size_dataset)
        self.return_index = return_index
        """
        The views that needs to be shuffled
        """
        self.shuffle_index = []
        shuffle = math.floor(shuffle_rate * len(self.percentage_crops))
        if shuffle > 0:
            self.shuffle_index = random.sample(range(len(self.percentage_crops)), shuffle)
        self.cache = {}

        """
        tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm],
                                                       selectable_pos=1)

        """
        checkpoint for the tokenizer
        """
        self.log_cnt = 0

        """
        add special tokens
        """
        if lm == 'roberta' or lm == 'bert':
            special_tokens_dict = {
                'additional_special_tokens': ["<subjectcol>", "<header>", "</subjectcol>", "</header>"]}
            self.header_token = ('<header>', '</header>')
            self.SC_token = ('<subjectcol>', '</subjectcol>')
        else:
            special_tokens_dict = {'additional_special_tokens': ["[subjectcol]", "[header],[/subjectcol],[/header]"]}
            self.header_token = ('[header]', '[/header]')
            self.SC_token = ('[subjectcol]', '[/subjectcol]')

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.special_tokens_map.items()
        self.max_len = max_length

    def _create_transforms(self):
        percentage_list = []
        trans = []
        for index, number in enumerate(self.nmb_crops):
            percentage_list.extend([self.percentage_crops[index]]*number)
        for i in range(len(percentage_list)):
            trans.append((percentage_list[i], self.augmentation_methods[i]))
        print(trans)
        return trans

    def _read_item(self, id):
        """Read a data item from the cache"""
        if id in self.cache:
            data = self.cache[id]
        else:
            if self.column is True:
                combination = self.samples[id].split("|")
                table, column = combination[0], combination[1]
                fn = os.path.join(self.path, table)
                column = pd.read_csv(fn)[column]
                data = pd.DataFrame(column)
            else:
                data = pd.read_csv(os.path.join(self.path, self.samples[id]))
        return data

    def _column_stratgy(self, table, max_tokens):
        def tokenize(text_ele):
            return text_ele.lower().split()
        col_texts = {}
        for index, column in enumerate(table.columns):
            column_values = table.iloc[:, index]
            all_text = column_values.tolist()
            # Use a set to store all unique tokens
            unique_tokens = set()
            # Iterate every text and tokenize the text, add it to the collection
            for text in all_text:
                tokens = tokenize(text)
                unique_tokens.update(tokens)
            unique_tokens = sorted(list(unique_tokens))
            string_token = ' '.join(unique_tokens[:max_tokens])
            col_text = self.tokenizer.cls_token + " "
            # value in column as a whole string mode
            if self.header:
                col_text += str(column) + " "
            # column value concatenating mode
            else:
                col_text += string_token + " "
            #col_texts[column] = col_text
            col_texts[index] = col_text
        return col_texts

    def _tokens(self, data: pd.DataFrame):
        """Tokenize a DataFrame table
            Args:
                data (DataFrame): the input TABLE/COLUMN dataframe
            Returns:
                List of int: list of token ID's with special tokens inserted
                Dictionary: a map from column names to special tokens
        """
        res = []
        max_tokens = self.max_len * 2 // len(data.columns) if len(data.columns) != 0 else 512
        budget = max(1, self.max_len // len(data.columns) - 1) if len(data.columns) != 0 else self.max_len
        # a map from column names to special token indices
        column_mp = {}
        # column-ordered preprocessing
        col_texts = self._column_stratgy(data, max_tokens)
        for column_index, col_text in col_texts.items():
            column_mp[column_index] = len(res)
            encoding = self.tokenizer.encode(text=col_text,
                                             max_length=budget,
                                             add_special_tokens=False,
                                             truncation=True)
            res += encoding
        self.log_cnt += 1
        #if self.log_cnt % 5000 == 0:
            #print(self.tokenizer.decode(res))
        return res, column_mp

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.column is True:
            combination = self.samples[index].split("|")
            table, column = combination[0], combination[1]
            fn = os.path.join(self.path, table)
            column = pd.read_csv(fn)[column]  # encoding="latin-1",
            data = pd.DataFrame(column)

        else:
            data = pd.read_csv(os.path.join(self.path, self.samples[index]))
        if self.subject_column is True:
            cols = subjectCol(data)
            if len(cols) > 0:
                data = data[cols]
        multi_crops = []
        for index, tuple_aug in enumerate(self.trans):
            percentage, aug_method = tuple_aug
            shuffle = False
            if index in self.shuffle_index:
                shuffle = True
            multi_crops.append(augment(data, aug_method, percent=percentage, shuffle=shuffle))
        multi_crop_tokens = [self._tokens(data) for data in multi_crops]
        x_values = [x for x, _ in multi_crop_tokens]
        mp_values = [mp for _, mp in multi_crop_tokens]
        cls_indices = []
        for col in mp_values[0]:
            # print(col, [mp[col] for mp in mp_values if col in mp])
            if all(col in mp for mp in mp_values):
                cls_indices.append(tuple(mp[col] for mp in mp_values))
        if self.return_index:
            return index, *x_values, cls_indices

        return *x_values, cls_indices

    def pad(self, batch):
        # Dynamically determine the number of sequences
        num_sequences = len(batch[0])
        sequences = list(zip(*batch))
        x_seqs = sequences[:-1]
        cls_indices = sequences[-1]
        # print(cls_indices)
        # Determine maximum length across all sequences
        maxlen = max([max([len(x) for x in seq]) for seq in x_seqs])

        # Pad the sequences
        x_new_seqs = []
        for seq in x_seqs:
            x_new = [xi + [self.tokenizer.pad_token_id] * (maxlen - len(xi)) for xi in seq]
            x_new_seqs.append(torch.LongTensor(x_new))
        # Decompose the column alignment
        if len(cls_indices[0]) == 0:
            per_cls_indices = [[] for _ in range(len(cls_indices))]
            cls_lists = per_cls_indices, per_cls_indices
        else:
            cls_lists = tuple([[] for _ in range(len(cls_indices[0][0]))])
            for table_batch in cls_indices:
                table_batch_indices = [[] for _ in range(len(cls_indices[0][0]))]
                for item in table_batch:
                    for i, idx in enumerate(item):
                        table_batch_indices[i].append(idx)
                for i, item in enumerate(table_batch_indices):
                    cls_lists[i].append(item)
        return *x_new_seqs, cls_lists
