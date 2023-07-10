import pickle
from argparse import Namespace
import numpy as np
import pandas as pd
import os
from torch.utils import data
from transformers import AutoTokenizer
from Utils import lm_mp

class Encoding(data.Dataset):
    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm],
                                                       selectable_pos=1)
        self.max_len = max_len
        self.path = path
        self.tables = [fn for fn in os.listdir(path) if '.csv' in fn]
        # only keep the first n tables
        self.table_cache = {}
    @staticmethod
    def from_hp(path: str, hp: Namespace):
        """Construct a PretrainTableDataset from hyperparameters

        Args:
            path (str): the path to the table directory
            hp (Namespace): the hyperparameters

        Returns:
            PretrainTableDataset: the constructed dataset
        """
        return Encoding(path,
                                    lm=hp.lm,
                                    max_len=hp.max_len,
                                    size=hp.size)
    def _read_table(self, table_id):
        """Read a table"""
        if table_id in self.table_cache:
            table = self.table_cache[table_id]
        else:
            fn = os.path.join(self.path, self.tables[table_id])
            table = pd.read_csv(fn, lineterminator='\n')  # encoding="latin-1",
        return table

    def _tokenize(self, table_name, table: pd.DataFrame):
        budget = max(1, self.max_len // len(table.columns) - 1) if len(table.columns) != 0 else self.max_len
        column_mp = []
        for index, column in enumerate(table.columns):

            encoding = self.tokenizer.encode(text=column,
                                             max_length=budget,
                                             add_special_tokens=False,
                                             truncation=True)
            column_mp.append(encoding)
        return (table_name, column_mp)
    def __len__(self):
        """Return the size of the dataset."""
        return len(self.tables)

    def allEmbeddings(self,store_path):
        all_encodings = []
        for idx in range(0,len(self.tables)):
            table = self._read_table(idx)
            table_name = self.tables[idx]
            table_encoding = self._tokenize(table_name, table)
            all_encodings.append(table_encoding)
        with open(store_path, 'wb') as fp:
            pickle.dump(all_encodings, fp)
        return all_encodings





