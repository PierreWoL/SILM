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

from transformers import AutoTokenizer

logger = getLogger()
from torch.utils.data import Dataset
import pandas as pd
from augment import augment

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}


class MultiCropTableDataset(Dataset):
    def __init__(self,
                 path,
                 percentage_crops: list,
                 size_dataset=-1,
                 shuffle_rate =0.2,
                 lm='roberta',
                 subject_column=False,
                 augmentation_methods="sample_cells_TFIDF",
                 column=False,
                 header = False,
                 return_index=False,
                 max_length = 512):



        self.path = path
        self.column = column
        self.header = header
        self.percentage_crops = percentage_crops
        self.augmentation_methods = augmentation_methods
        """
        Transfer augmentation methods list
        """
        if isinstance(augmentation_methods, str):
            self.augmentation_methods = [augmentation_methods] * len(percentage_crops)
            print(self.augmentation_methods)
        else:
            self.augmentation_methods = augmentation_methods
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
            self.samples = self.samples[:size_dataset]
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
        self.log_cnt =0


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


    def _create_transforms(self):
        trans = []
        for index in range(len(self.percentage_crops)):
            trans.append((self.percentage_crops[index], self.augmentation_methods[index]))
            print(trans)
        return trans

    def  _tokens(self, data:pd.DataFrame):
        """Tokenize a DataFrame table
            Args:
                table (DataFrame): the input TABLE/COLUMN dataframe
            Returns:
                List of int: list of token ID's with special tokens inserted
                Dictionary: a map from column names to special tokens
        """
        res = []
        max_tokens = self.max_len * 2 // len(data.columns) if len(data.columns) != 0 else 512
        budget = max(1, self.max_len // len(data.columns) - 1) if len(data.columns) != 0 else self.max_len
        tfidfDict = computeTfIdf(table) if "tfidf" in self.sample_meth else None  # from preprocessor.py
        # a map from column names to special token indices
        column_mp = {}
        # column-ordered preprocessing
        col_texts = self._column_stratgy(Sub_cols_header, table, tfidfDict, max_tokens)
        for column, col_text in col_texts.items():
            column_mp[column] = len(res)
            encoding = self.tokenizer.encode(text=col_text,
                                             max_length=budget,
                                             add_special_tokens=False,
                                             truncation=True)
            res += encoding
        self.log_cnt += 1
        if self.log_cnt % 5000 == 0:
            print(self.tokenizer.decode(res))
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
            print(data)
        else:
            data = pd.read_csv(os.path.join(self.path, self.samples[index]))
        multi_crops = []
        for index, percentage, aug_method in enumerate(self.trans):
            shuffle = False
            if index in self.shuffle_index:
                shuffle = True
            multi_crops.append(augment(data, aug_method, percent=percentage, shuffle=shuffle))


        if self.return_index:
            return index, multi_crops
        return multi_crops
