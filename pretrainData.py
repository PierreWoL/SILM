from argparse import Namespace
import torch
import random
import pandas as pd
import os
import TableAnnotation as TA
from torch.utils import data
from transformers import AutoTokenizer
from starmie.sdd.augment import augment
from typing import List
from starmie.sdd.preprocessor import computeTfIdf, tfidfRowSample, preprocess
from SubjectColumnDetection import ColumnType
from d3l.utils.functions import token_stop_word

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}


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
                 check_subject_Column='subjectheader'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm],
                                                       selectable_pos=1)  # , additional_special_tokens=['[header]', '[SC]']

        if lm == 'roberta':
            special_tokens_dict = {'additional_special_tokens': ["<sc>", "<header>", "</sc>", "</header>"]}
            self.header_token = ('<header>', '</header>')
            self.SC_token = ('<sc>', '</sc>')
        else:
            special_tokens_dict = {'additional_special_tokens': ["[SC]", "[HEADER],[SCE],[HEADERE]"]}
            self.header_token = ('[HEADER]', '[HEADERE]')
            self.SC_token = ('"[SC]"', '[SCE]')
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        special_tokens = self.tokenizer.special_tokens_map.items()

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
                                    check_subject_Column=hp.check_subject_Column)

    def _read_table(self, table_id):
        """Read a table"""
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn, lineterminator='\n')  # encoding="latin-1",
            self.table_cache[table_id] = table
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
        # print(table.transpose(),table.shape)
        max_tokens = self.max_len * 2 // len(table.columns)
        budget = max(1, self.max_len // len(table.columns) - 1)
        # print("max_tokens,budget",max_tokens,budget)
        tfidfDict = computeTfIdf(table) if "tfidf" in self.sample_meth else None  # from preprocessor.py
        # print(table.transpose())
        # print(tfidfDict)
        # a map from column names to special token indices
        column_mp = {}
        Sub_cols_header = []
        if 'subject' in self.check_subject_Column:
            if self.tables[idx] in [fn for fn in os.listdir(self.path[:-4] + "SubjectColumn") if
                                    '.csv' in fn]:
                Sub_cols = pd.read_csv(self.path[:-4] + "SubjectColumn/" + self.tables[idx],
                                       )  # encoding="latin-1"
                Sub_cols_header = Sub_cols.columns.tolist()
            else:
                anno = TA.TableColumnAnnotation(table)
                types = anno.annotation
                # print(types)
                for key, type in types.items():
                    if type == ColumnType.named_entity:
                        Sub_cols_header = [table.columns[key]]
                        break
        # column-ordered preprocessing
        if self.table_order == 'column':
            if 'row' in self.sample_meth:
                table = tfidfRowSample(table, tfidfDict, max_tokens)
                # print("table after tfidf row sample \n", table)
            for column in table.columns:
                tokens = preprocess(table[column], tfidfDict, max_tokens, self.sample_meth)  # from preprocessor.py
                string_token = ' '.join(tokens[:max_tokens])
                col_text = self.tokenizer.cls_token + " "
                if 'header' in self.check_subject_Column:
                    col_text += self.header_token[0] + " " + column + " " + self.header_token[1] + " "  #

                if 'subject' in self.check_subject_Column and column in Sub_cols_header:

                    col_text += self.SC_token[0] + " " + string_token + " " + self.SC_token[1] + " "  #
                else:
                    col_text += string_token + " "  # string_token + " "
                # print(col_text)
                column_mp[column] = len(res)
                encoding = self.tokenizer.encode(text=col_text,
                                                 max_length=budget,
                                                 add_special_tokens=False,
                                                 truncation=True)
                res += encoding
        if 'row' in self.table_order:
            max_tokens = self.max_len * 2  # // len(table)
            budget = self.max_len  # max(1, self.max_len // len(table) - 1)
            table_text = self.tokenizer.cls_token + " "
            for index, row in table.iterrows():
                if index == 0:
                    if "pure" in self.table_order and 'header' in self.check_subject_Column:
                        header_text = self.tokenizer.cls_token + " " + \
                                      self.header_token[0] + " " + " ".join(table.columns) + \
                                      " " + self.header_token[1] + " "
                        budget = self.max_len - len(header_text)
                        # print("header", header_text)
                        column_mp[index] = len(res)
                        encoding = self.tokenizer.encode(text=header_text,
                                                         max_length=budget,
                                                         add_special_tokens=False,
                                                         truncation=True)
                        res += encoding
                        continue

                row_text = ""
                for column, cell in row.items():
                    token_cell = str(" ".join(token_stop_word(cell)))
                    cell_token = str(column) + " " + token_cell if 'sentence' in self.table_order else token_cell
                    row_text += self.SC_token[0] + " " + cell_token + " " + self.SC_token[1] + " " \
                        if column in Sub_cols_header else cell_token + " "
                if len(table_text.split(" ")) > max_tokens:
                    print(table_text)
                    encoding = self.tokenizer.encode(text=table_text,
                                                     max_length=budget,
                                                     add_special_tokens=False,
                                                     truncation=True)
                    column_mp[index] = len(res)
                    res += encoding
                    break
                else:
                    table_text += row_text + self.tokenizer.sep_token + " "
        self.log_cnt += 1
        if self.log_cnt % 5000 == 0:
            print(self.tokenizer.decode(res))
        # print(len(res), len(column_mp))
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
        # print("table_ori",len(table_ori))
        # single-column mode: only keep one random column
        if "row" in self.table_order:
            tfidfDict = computeTfIdf(table_ori)
            table_ori = tfidfRowSample(table_ori, tfidfDict, 0)
            print("table_ori", table_ori.T )
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
        # add in here!!!!!!!!!!
        if "pure" in self.table_order and 'header' in self.check_subject_Column:
            header = table_ori.columns.tolist()
            table_ori = pd.DataFrame([header] + table_ori.values.tolist(), columns=header)
            table_aug = pd.DataFrame([header] + table_aug.values.tolist(), columns=header)
        x_ori, mp_ori = self._tokenize(table_ori, idx=idx)
        """
         if "row" in self.table_order:
            table_ori = table_ori.iloc[:len(mp_ori), :]
            table_aug = table_aug.iloc[:len(mp_ori), :]
        """
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
