import pickle
from argparse import Namespace

import numpy as np
import torch
import random
import pandas as pd
import os

from sentence_transformers import SentenceTransformer
from Utils import subjectCol
import TableAnnotation as TA
from torch.utils import data
from transformers import AutoTokenizer, AutoModel, AutoConfig
from starmie.sdd.augment import augment
from typing import List
from starmie.sdd.preprocessor import computeTfIdf, tfidfRowSample, preprocess
from SubjectColumnDetection import ColumnType
import d3l.utils.functions as fun
from Utils import aug

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
                 subject_column=False,
                 header=False,
                 pretrain=False,
                 sample_meth='wordProb',
                 table_order='column',
                 check_subject_Column='subjectheader'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm],
                                                       selectable_pos=1)
        # pretained-LM
        self.pretrain = pretrain
        self.lm = lm
        self.model = None

        if lm == 'roberta':
            special_tokens_dict = {
                'additional_special_tokens': ["<subjectcol>", "<header>", "</subjectcol>", "</header>"]}
            self.header_token = ('<header>', '</header>')
            self.SC_token = ('<subjectcol>', '</subjectcol>')

            if self.pretrain:
                print("ROBERTa")
                self.model = AutoModel.from_pretrained(lm_mp[lm])
                tokenizer_vocab_size = self.tokenizer.vocab_size + len(special_tokens_dict['additional_special_tokens'])
                self.model.resize_token_embeddings(new_num_tokens=tokenizer_vocab_size)

        else:
            special_tokens_dict = {'additional_special_tokens': ["[subjectcol]", "[header],[/subjectcol],[/header]"]}
            self.header_token = ('[header]', '[/header]')
            self.SC_token = ('[subjectcol]', '[/subjectcol]')
            if self.pretrain:
                print("SentenceTransformer")
                self.model = SentenceTransformer(lm_mp[lm])

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.special_tokens_map.items()
        self.max_len = max_len
        self.path = path

        # assuming tables are in csv format
        self.subjectColumn_path = os.path.join(self.path[:-4], "SubjectColumn")
        self.isCombine = False
        if "TabFact" in self.path:
            self.subjectColumn_path = False
            self.isCombine = True

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

        # subject-column mode
        self.subject_column = subject_column

        # header-only  mode
        self.header = header

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
                                    subject_column=hp.subject_column,
                                    sample_meth=hp.sample_meth,
                                    table_order=hp.table_order,
                                    header=hp.header,
                                    pretrain=hp.pretrain,
                                    check_subject_Column=hp.check_subject_Column)

    def _read_table(self, table_id):
        """Read a table"""
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn)  # encoding="latin-1",
            if self.isCombine:
                table = table.iloc[:, 1:]  # encoding="latin-1",

            self.table_cache[table_id] = table
        return table

    def _tokenize(self, table: pd.DataFrame):  # -> List[int]
        """Tokenize a DataFrame table

        Args:
            table (DataFrame): the input table

        Returns:
            List of int: list of token ID's with special tokens inserted
            Dictionary: a map from column names to special tokens
        """
        res = []
        max_tokens = self.max_len * 2 // len(table.columns) if len(table.columns) != 0 else 512
        budget = max(1, self.max_len // len(table.columns) - 1) if len(table.columns) != 0 else self.max_len
        tfidfDict = computeTfIdf(table,isCombine=self.isCombine) if "tfidf" in self.sample_meth else None  # from preprocessor.py

        # a map from column names to special token indices
        column_mp = {}
        Sub_cols_header = subjectCol(table, self.isCombine)
        # column-ordered preprocessing
        if self.table_order == 'column':
            col_texts = self._column_stratgy(Sub_cols_header, table, tfidfDict, max_tokens)
           
            for column, col_text in col_texts.items():
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
                        max_tokens = self.max_len * 2 // len(table)  # // len(table)
                        budget = max(1, self.max_len)
                        column_mp[index] = len(res)
                        encoding = self.tokenizer.encode(text=header_text,
                                                         max_length=budget,
                                                         add_special_tokens=False,
                                                         truncation=True)
                        res += encoding
                        continue

                row_text = ""
                for column, cell in row.items():
                    token_cell = str(" ".join(fun.token_stop_word(cell)))
                    cell_token = str(column) + " " + token_cell if 'sentence' in self.table_order else token_cell
                    row_text += self.SC_token[0] + " " + cell_token + " " + self.SC_token[1] + " " \
                        if column in Sub_cols_header else cell_token + " "
                if len(table_text.split(" ")) > max_tokens:
                    break
                else:
                    table_text += row_text + self.tokenizer.sep_token + " "
                encoding = self.tokenizer.encode(text=table_text,
                                                 max_length=budget,
                                                 add_special_tokens=False,
                                                 truncation=True)
                column_mp[index] = len(res)
                res += encoding

        self.log_cnt += 1
        if self.log_cnt % 5000 == 0:
            print(self.tokenizer.decode(res))
        return res, column_mp

    def _column_stratgy(self, Sub_cols_header, table, tfidfDict, max_tokens, NoToken=False):
        col_texts = {}
        if 'row' in self.sample_meth:
            table = tfidfRowSample(table, tfidfDict, max_tokens)
        for index, column in enumerate(table.columns):
            column_values = table.iloc[:, index] if self.isCombine is False \
                else pd.Series(table.iloc[:, index][0].split(",")).rename(column)
            tokens = preprocess(column_values, tfidfDict, max_tokens, self.sample_meth)  # from preprocessor.py
            string_token = ' '.join(tokens[:max_tokens])
            col_text = self.tokenizer.cls_token + " "
            # value in column as a whole string mode
            if NoToken is False:
                # header-only mode
                if self.header:
                    if 'subject' in self.check_subject_Column:
                        col_text += self.SC_token[0] + " " + str(column) + " " + self.SC_token[1] + " "  #
                    else:
                        col_text += str(column) + " "
                # column value concatenating mode
                else:
                    if 'header' in self.check_subject_Column:
                        col_text += self.header_token[0] + " " + str(column) + " " + self.header_token[1] + " "  #

                    if 'subject' in self.check_subject_Column and column in Sub_cols_header:

                        col_text += self.SC_token[0] + " " + string_token + " " + self.SC_token[1] + " "  #
                    else:
                        col_text += string_token + " "

                col_texts[column] = col_text
            # average embedding of tokens mode
            else:
                column_token = fun.token_list(fun.remove_blank(column_values))
                if column_token != None:
                    col_texts[column] = column_token
                else:
                    list_values = column_values.tolist()
                    col_texts[column] = [str(i) for i in list_values]
        return col_texts

    def _encode(self, table: pd.DataFrame, Token=False):
        max_tokens = self.max_len * 2 // len(table.columns) if len(table.columns) != 0 else 512
        tfidfDict = computeTfIdf(table) if "tfidf" in self.sample_meth else None  # from preprocessor.py
        budget = max(1, self.max_len // len(table.columns) - 1) if len(table.columns) != 0 else self.max_len
        # a map from column names to special token indices
        Sub_cols_header = subjectCol(table, self.isCombine)
        embeddings = []
        # column-ordered preprocessing
        if self.table_order == 'column':
            col_texts = self._column_stratgy(Sub_cols_header, table, tfidfDict, max_tokens, NoToken=Token)
            print(col_texts)
            for column, col_text in col_texts.items():
                if self.lm == "sbert":
                    embedding = self.model.encode(col_text)
                    if Token is False:
                        embeddings.append(embedding)
                    else:
                        average = np.mean(embedding, axis=0)
                        embeddings.append(average)

                elif self.lm == "roberta":
                    if Token is False:
                        tokens = self.tokenizer.encode_plus(col_text, add_special_tokens=True, max_length=512,
                                                            truncation=True, return_tensors="pt")
                        # Perform the encoding using the model
                        with torch.no_grad():
                            outputs = self.model(**tokens)
                        # Extract the last hidden state (embedding) from the outputs
                        last_hidden_state = outputs.last_hidden_state.mean(dim=1)[0]
                        embeddings.append(last_hidden_state)
                    else:
                        embeddings_per_col = []
                        for text in col_text:
                            tokens = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=512,
                                                                truncation=True, return_tensors="pt")
                            with torch.no_grad():
                                outputs = self.model(**tokens)
                            last_hidden_state = outputs.last_hidden_state.mean(dim=1)[0]
                            embeddings_per_col.append(last_hidden_state)
                        stacked_embeddings = torch.stack(embeddings_per_col)
                        average_encoding = torch.mean(stacked_embeddings, dim=0)

                        embeddings.append(average_encoding)

        return embeddings

    def encodings(self, output_path, setting=False):
        table_encodings = []
        for idx in range(len(self.tables)):
            fn = os.path.join(self.path, self.tables[idx])
            table_ori = pd.read_csv(fn)
            if "row" in self.table_order:
                tfidfDict = computeTfIdf(table_ori)
                table_ori = tfidfRowSample(table_ori, tfidfDict, 0)
            if self.single_column:
                col = random.choice(table_ori.columns)
                table_ori = table_ori[[col]]
            if self.subject_column:
                cols = subjectCol(table_ori, self.isCombine)
                if len(cols) > 0:
                    table_ori = table_ori[cols]
            embedding = self._encode(table_ori, Token=setting)
            table_encodings.append((self.tables[idx], np.array(embedding)))

        output_file = "Pretrain_%s_%s_%s_%s_%s.pkl" % (self.lm, self.sample_meth,
                                                       self.table_order, self.check_subject_Column, setting)
        if self.single_column:
            output_file = "Pretrain_%s_%s_%s_%s_%s_singleCol.pkl" % (self.lm, self.sample_meth,
                                                                     self.table_order, self.check_subject_Column,
                                                                     setting)
        if self.subject_column:
            output_file = "Pretrain__%s_%s_%s_%s_%s_subCol.pkl" % (self.lm, self.sample_meth,
                                                                   self.table_order, self.check_subject_Column, setting)

        if self.header:
            output_file = "Pretrain_%s_%s_%s_%s_%s_header.pkl" % (self.lm, self.sample_meth,
                                                                  self.table_order, self.check_subject_Column, setting)

        target_path = os.path.join(output_path, output_file)
        pickle.dump(table_encodings, open(target_path, "wb"))
        return table_encodings

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
        # single-column mode: only keep one random column
        if "row" in self.table_order:
            tfidfDict = computeTfIdf(table_ori)
            table_ori = tfidfRowSample(table_ori, tfidfDict, 0)

        if self.single_column:
            col = random.choice(table_ori.columns)
            table_ori = table_ori[[col]]
        if self.subject_column:
            cols = subjectCol(table_ori, self.isCombine)
            if len(cols) > 0:
                table_ori = table_ori[cols]
            """if os.path.exists(self.subjectColumn_path):
                subcol_files = [fn for fn in os.listdir(self.path[:-4] + "SubjectColumn") if
                    '.csv' in fn]
                if self.tables[idx] in subcol_files:
                    Sub_cols = pd.read_csv(self.path[:-4] + "SubjectColumn/" + self.tables[idx])
                    cols = [col for col in Sub_cols.columns if col in table_ori.columns]
            else:
            """

        # apply the augmentation operator
        if ',' in self.augment_op:
            op1, op2 = self.augment_op.split(',')
            table_tmp = table_ori
            table_ori = augment(table_tmp, op1)
            table_aug = augment(table_tmp, op2)
        else:
            table_aug = augment(table_ori, self.augment_op)

            if self.isCombine:
                table_aug = aug(table_ori)

        if "pure" in self.table_order and 'header' in self.check_subject_Column:
            header = table_ori.columns.tolist()
            table_ori = pd.DataFrame([header] + table_ori.values.tolist(), columns=header)
            table_aug = pd.DataFrame([header] + table_aug.values.tolist(), columns=header)
        x_ori, mp_ori = self._tokenize(table_ori)
        x_aug, mp_aug = self._tokenize(table_aug)

        cls_indices = []
        for col in mp_ori:
            if col in mp_aug:
                cls_indices.append((mp_ori[col], mp_aug[col]))
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
        for item in cls_indices:
            cls_ori.append([])
            cls_aug.append([])

            for idx1, idx2 in item:
                cls_ori[-1].append(idx1)
                cls_aug[-1].append(idx2)

        return torch.LongTensor(x_ori_new), torch.LongTensor(x_aug_new), (cls_ori, cls_aug)
