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
                 check_subject_Column='subjectheader'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm],
                                                       selectable_pos=1)  # , additional_special_tokens=['[header]', '[SC]']

        special_tokens = self.tokenizer.special_tokens_map.items()

        if lm == 'roberta':
            special_tokens_dict = {'additional_special_tokens': ["<sc>", "<header>", "<s/c>", "</header>"]}
            self.header_token = ('<header>', '</header>')
            self.SC_token = ('<sc>', '</sc>')
        else:
            special_tokens_dict = {'additional_special_tokens': ["[SC]", "[HEADER],[SCE],[HEADERE]"]}
            self.header_token = ('[HEADER]', '[HEADERE]')
            self.SC_token = ('"[SC]"', '[SCE]')
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        for token_name, token_id in special_tokens:
            print(f"{token_name}: {token_id}")
        """
           
        original_id_to_token = dict(zip(original_vocab.values(), original_vocab.keys()))
        new_id_to_token = {}
        new_id_to_token_backup = {}
        new_id_to_token[0] = '<s>'
        new_id_to_token[1] = '<header>'
        new_id_to_token[2] = '</header>'
        new_id_to_token[3] = '<sc>'
        new_id_to_token[4] = '</sc>'
        for i in range(1, len(original_vocab) - 1):
            new_id_to_token[i + 4] = original_id_to_token[i]
        new_token_to_id = {v: k for k, v in new_id_to_token.items()}
        print(new_token_to_id)
        self.tokenizer.save_vocabulary("vocab.txt", new_vocab=new_token_to_id)
         header_token = '<header>'
        subject_column_token = '<sc>'
        end_subject_column_token = '<\sc>'
        # Add the special tokens to the tokenizer
        special_tokens_dict = {
            'additional_special_tokens': [header_token, subject_column_token, end_subject_column_token]}
        #print(special_tokens_dict)
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        """
        # additional_special_tokens = ['<header>','<sc>']
        # additional_special_tokens = {'additional_special_tokens': ['<header>','<sc>']}
        # self.tokenizer.add_special_tokens(additional_special_tokens)
        """
        sc_token_id = self.tokenizer.convert_tokens_to_ids('<sc>')
        assert sc_token_id == 1
        sc_token_id = self.tokenizer.convert_tokens_to_ids('<header>')
        assert sc_token_id == 2
        """

        # print("token is" ,self.tokenizer.get_vocab()[self.tokenizer.cls_token], self.tokenizer.get_vocab()['<header>'],self.tokenizer.get_vocab()['<sc>'])
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
            table = pd.read_csv(fn, encoding="latin-1")  # lineterminator='\n'
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
        # print(table.transpose())
        max_tokens = self.max_len * 2 // len(table.columns)
        budget = max(1, self.max_len // len(table.columns) - 1)
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
                                       encoding="latin-1")
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
                if len(string_token) >= 500:
                    string_token = string_token[:500]
                col_text = self.tokenizer.cls_token + " "
                if 'header' in self.check_subject_Column:
                    #  col_text += column + " " + self.tokenizer.sep_token + " " + self.tokenizer.cls_token + " "
                    col_text += self.header_token[0] + " " + column + " " + self.header_token[1] + " "
                    column_mp[column] = len(res)
                    if 'subject' in self.check_subject_Column and column in Sub_cols_header:
                        # col_text += self.tokenizer.mask_token+ " " + string_token + " "
                        col_text += self.SC_token[0] + " " + string_token + " " + self.SC_token[1] + " "
                    else:
                        col_text += string_token + " "  # string_token + " "
                else:
                    col_text += string_token + " "
                print(col_text)
                column_mp[column] = len(res)
                encoding = self.tokenizer.encode(text=col_text,
                                                 max_length=budget,
                                                 add_special_tokens=False,
                                                 truncation=True)
                res += encoding
                # print(len(res))





        else:
            # row-ordered preprocessing
            reached_max_len = False
            for rid in range(len(table)):
                row = table.iloc[rid:rid + 1]
                # print(row)
                for column in table.columns:
                    col_text = ''
                    tokens = preprocess(row[column], tfidfDict, max_tokens, self.sample_meth)  # from preprocessor.py
                    if rid == 0:
                        column_mp[column] = len(res)
                        if 'header' in self.check_subject_Column:
                            header_text = self.tokenizer.cls_token + " " + '<header>' + " "
                            if 'subject' in self.check_subject_Column and column in Sub_cols_header:
                                header_text += '<sc>' + column + " " + '<\sc>' + " "
                                col_text = self.tokenizer.pad_token + " " + '<sc>' + \
                                           ' '.join(tokens[:max_tokens]) + " " + '<\sc>' + " "
                            else:
                                header_text += column + " "
                                col_text = self.tokenizer.pad_token + " " + \
                                           ' '.join(tokens[:max_tokens]) + " "

                            tokenized = self.tokenizer.encode(text=header_text,
                                                              max_length=budget,
                                                              add_special_tokens=False,
                                                              truncation=True)
                            res += tokenized
                            # print(header_text)#, tokenized, res

                        else:
                            if 'subject' in self.check_subject_Column and column in Sub_cols_header:
                                col_text = self.tokenizer.cls_token + " " + \
                                           '<sc>' + ' '.join(tokens[:max_tokens]) + '<\sc>' + " "
                            else:
                                col_text = self.tokenizer.cls_token + " " + \
                                           ' '.join(tokens[:max_tokens]) + " "
                    else:
                        if 'subject' in self.check_subject_Column and column in Sub_cols_header:
                            col_text = self.tokenizer.pad_token + " " + \
                                       '<sc>' + ' '.join(tokens[:max_tokens]) + '<\sc>' + " "
                        else:
                            col_text = self.tokenizer.pad_token + " " + \
                                       ' '.join(tokens[:max_tokens]) + " "

                    tokenized = self.tokenizer.encode(text=col_text,
                                                      max_length=budget,
                                                      add_special_tokens=False,
                                                      truncation=True)

                    if len(tokenized) + len(res) <= self.max_len:
                        res += tokenized
                        # print(col_text)#,tokenized,res
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
        # print(table_ori)
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
