from argparse import Namespace
import torch
import random
import pandas as pd
import os
import TableAnnotation as TA
from torch.utils import data
from transformers import AutoTokenizer, BertTokenizer
from starmie.sdd.augment import augment
from typing import List
from starmie.sdd.preprocessor import computeTfIdf, tfidfRowSample, preprocess
from SubjectColumnDetection import ColumnType

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased'}


class PretrainTableDataset(data.Dataset):
    """Table dataset for pre-training"""

    def __init__(self,
                 path,
                 augment_op,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 single_column=False,
                 sample_meth='wordProb',
                 table_order='column',
                 check_subject_Column = 'subjectheader'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])  # , additional_special_tokens=['[header]', '[SC]']
        # additional_special_tokens = ['<header>', '<SC>']
        # self.tokenizer.add_tokens(additional_special_tokens)
        header_token = '<header>'
        subject_column_token = '<sc>'
        # Add the special tokens to the tokenizer
        special_tokens_dict = {'additional_special_tokens': [header_token, subject_column_token]}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.max_len = max_len
        self.path = path

        # assuming tables are in csv format
        self.tables = [fn for fn in os.listdir(path) if '.csv' in fn]

        # only keep the first n tables
        if size is not None:
            self.tables = self.tables[:size]

        self.table_cache = {}
        self.check_subject_Column = check_subject_Column
        # augmentation operators
        self.augment_op = augment_op

        # logging counter
        self.log_cnt = 0

        # sampling method
        self.sample_meth = sample_meth

        # single-column mode
        self.single_column = single_column

        # row or column order for preprocessing
        self.table_order = table_order

        # tokenizer cache
        self.tokenizer_cache = {}

    @staticmethod
    def from_hp(path: str, hp: Namespace):
        """Construct a PretrainTableDataset from hyperparameters

        Args:
            path (str): the path to the table directory
            hp (Namespace): the hyperparameters

        Returns:
            PretrainTableDataset: the constructed dataset
        """
        return PretrainTableDataset(path,
                                    augment_op=hp.augment_op,
                                    lm=hp.lm,
                                    max_len=hp.max_len,
                                    size=hp.size,
                                    single_column=hp.single_column,
                                    sample_meth=hp.sample_meth,
                                    table_order=hp.table_order,
                                    check_subject_Column = hp.check_subject_Column)

    def _read_table(self, table_id):
        """Read a table"""
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn, lineterminator='\n', encoding="latin-1")
            self.table_cache[table_id] = table
        # print(table_id, table)

        return table

    def _tokenize(self, table: pd.DataFrame, idx=-1):  # -> List[int]
        """Tokenize a DataFrame table

        Args:
            table (DataFrame): the input table

        Returns:
            List of int: list of token ID's with special tokens inserted
            Dictionary: a map from column names to special tokens
        """
        res = []
        max_tokens = self.max_len * 2 // len(table.columns)
        budget = max(1, self.max_len // len(table.columns) - 1)
        # print("max tokens and buget",max_tokens,budget)
        tfidfDict = computeTfIdf(table) if "tfidf" in self.sample_meth else None  # from preprocessor.py
        # print(table)
        # print(tfidfDict)
        # a map from column names to special token indices
        column_mp = {}
        Sub_cols_header = []
        # column-ordered preprocessing
        if self.table_order == 'column':

            if 'row' in self.sample_meth:
                table = tfidfRowSample(table, tfidfDict, max_tokens)
                # print("table after tfidf row sample \n", table)
            if 'subject' in self.check_subject_Column:
                # TODO  we need to think where to put the subject column detection, in get_item or in the tokenize
                if 'subject' in self.check_subject_Column:
                    if self.tables[idx] in [fn for fn in os.listdir(self.path[:-4] + "SubjectColumn") if '.csv' in fn]:
                        Sub_cols = pd.read_csv(self.path[:-4] + "SubjectColumn/" + self.tables[idx],
                                               encoding="latin-1")
                        original = pd.read_csv(self.path + "/" + self.tables[idx],
                                               encoding="latin-1")
                        Sub_cols_header = Sub_cols.columns.tolist()
                    else:
                        anno = TA.TableColumnAnnotation(table)
                        types = anno.annotation
                        print(types)
                        for key, type in types.items():
                            if type == ColumnType.named_entity:
                                Sub_cols_header = [table.columns[key]]
                                break

            for column in table.columns:
                tokens = preprocess(table[column], tfidfDict, max_tokens, self.sample_meth)  # from preprocessor.py
                if 'header' in self.check_subject_Column:
                    col_text = self.tokenizer.cls_token + " "+ '<header> ' + column+" "
                if 'subject' in self.check_subject_Column:
                    if column in Sub_cols_header:
                        col_text += '<sc>'+ " "
                col_text += ' '.join(tokens[:max_tokens]) + " "
                # print(col_text)
                column_mp[column] = len(res)
                print(col_text,column_mp,column)
                encoding = self.tokenizer.encode(text=col_text,
                                                 max_length=budget,
                                                 add_special_tokens=False,
                                                 truncation=True)
                res += encoding




        else:
            # row-ordered preprocessing
            reached_max_len = False
            for rid in range(len(table)):
                row = table.iloc[rid:rid + 1]
                for column in table.columns:
                    tokens = preprocess(row[column], tfidfDict, max_tokens, self.sample_meth)  # from preprocessor.py
                    if rid == 0:
                        column_mp[column] = len(res)
                        col_text = self.tokenizer.cls_token + " " + \
                                   ' '.join(tokens[:max_tokens]) + " "
                    else:
                        col_text = self.tokenizer.pad_token + " " + \
                                   ' '.join(tokens[:max_tokens]) + " "

                    tokenized = self.tokenizer.encode(text=col_text,
                                                      max_length=budget,
                                                      add_special_tokens=False,
                                                      truncation=True)

                    if len(tokenized) + len(res) <= self.max_len:
                        res += tokenized
                    else:
                        reached_max_len = True
                        break

                if reached_max_len:
                    break

        self.log_cnt += 1
        if self.log_cnt % 5000 == 0:
            print(self.tokenizer.decode(res))
        # print(res, column_mp)
        return res, column_mp

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.tables)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the first view
            List of int: token ID's of the second view
        """
        table_ori = self._read_table(idx)
        print(table_ori)
        # single-column mode: only keep one random column
        if self.single_column:
            col = random.choice(table_ori.columns)
            table_ori = table_ori[[col]]

        # apply the augmentation operator
        if ',' in self.augment_op:
            op1, op2 = self.augment_op.split(',')
            table_tmp = table_ori
            table_ori = augment(table_tmp, op1)
            table_aug = augment(table_tmp, op2)
        else:
            table_aug = augment(table_ori, self.augment_op)

        # convert table into string
        x_ori, mp_ori = self._tokenize(table_ori, idx=idx)
        x_aug, mp_aug = self._tokenize(table_aug, idx=idx)

        # make sure that x_ori and x_aug has the same number of cls tokens
        # x_ori_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_ori])
        # x_aug_cnt = sum([int(x == self.tokenizer.cls_token_id) for x in x_aug])
        # assert x_ori_cnt == x_aug_cnt

        # insertsect the two mappings
        cls_indices = []
        for col in mp_ori:
            if col in mp_aug:
                cls_indices.append((mp_ori[col], mp_aug[col]))
                # print(cls_indices)

        return x_ori, x_aug, cls_indices

    def pad(self, batch):
        """Merge a list of dataset items into a training batch

        Args:
            batch (list of tuple): a list of sequences

        Returns:
            LongTensor: x_ori of shape (batch_size, seq_len)
            LongTensor: x_aug of shape (batch_size, seq_len)
            tuple of List: the cls indices
        """
        x_ori, x_aug, cls_indices = zip(*batch)
        # print(cls_indices)
        max_len_ori = max([len(x) for x in x_ori])
        max_len_aug = max([len(x) for x in x_aug])
        maxlen = max(max_len_ori, max_len_aug)
        x_ori_new = [xi + [self.tokenizer.pad_token_id] * (maxlen - len(xi)) for xi in x_ori]
        x_aug_new = [xi + [self.tokenizer.pad_token_id] * (maxlen - len(xi)) for xi in x_aug]

        # decompose the column alignment
        cls_ori = []
        cls_aug = []
        # print("cls_indices is ",cls_indices)
        for item in cls_indices:
            cls_ori.append([])
            cls_aug.append([])

            for idx1, idx2 in item:
                cls_ori[-1].append(idx1)
                cls_aug[-1].append(idx2)
                # print("cls_ori and cls_aug   ", cls_ori, cls_aug)

        return torch.LongTensor(x_ori_new), torch.LongTensor(x_aug_new), (cls_ori, cls_aug)
