# Copyright 2023 Jiaoyan Chen. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
import pickle
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @paper(
#     "Contextual Semantic Embeddings for Ontology Subsumption Prediction (World Wide Web Journal)",
# )
from datasets import Dataset
from argparse import Namespace
from Dataset_dict import dataframe_train, create_column_pairs_mapping
import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}


def save_dict(dataset_dict, path):
    with open(path, "wb") as f:
        pickle.dump(dataset_dict, f)

class ColumnMatchingClassifier:
    def __init__(self,
                 train_path,
                 positive_op,
                 max_len=256,
                 lm='roberta',
                 early_stop: bool = False,
                 early_stop_patience: int = 10):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm], selectable_pos=1)
        self.model = AutoModelForSequenceClassification.from_pretrained(lm_mp[lm])
        self.positive_op = positive_op
        self.trainer = None
        self.mapping = []
        self.train_path = train_path
        if any(file.endswith('.pkl') for file in os.listdir(train_path)):
            with open(self.train_path+"/dataset_dict.pkl", 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            dfs = dataframe_train(self.train_path)
            self.dataset = create_column_pairs_mapping(dfs, positive_op)[0]
            #print(self.dataset,type(self.dataset),type(self.dataset["train"]))
            """TODO this may need specified path"""
            save_dict(self.dataset, self.train_path + "/dataset_dict.pkl")
        
        self.train_data = self.load_dataset(self.dataset["train"], max_length=2*max_len)
        self.eval_data = self.load_dataset(self.dataset["eval"], max_length=2*max_len)
        
        print("train and eval datasets are ",self.train_data,self.eval_data)
        self.max_length = max_len
        self.tables = [fn for fn in os.listdir(train_path) if '.csv' in fn]

        # methods to create positive samples
        print(f"data files loaded with sizes:")
        print(f"\t[# Train]: {len(self.train_data)}")

        # early stopping
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience

    @staticmethod
    def from_hp(path: str, hp: Namespace):
        """Construct Column matching from hyperparameters
        Args:
            path (str): the path to the table directory
            hp (Namespace): the hyperparameters
        Returns:
            ColumnMatching: the constructed column pairs from different tables
        """
        return ColumnMatchingClassifier(path,
                                        positive_op=hp.positive_op,
                                        lm=hp.lm,
                                        max_len=hp.max_len,
                                        early_stop=hp.early_stop,
                                        early_stop_patience=hp.early_stop_patience
                                        )

    "slice or random select the cells from the particular columns"

    def train(self, train_args: TrainingArguments, do_fine_tune: bool = True):
        r"""Initiate the Huggingface trainer with input arguments and start training.
        Args:
            train_args (TrainingArguments): Arguments for training.
            do_fine_tune (bool): `False` means loading the checkpoint without training. Defaults to `True`.
        """
        self.trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        if self.early_stop:
            self.trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=self.early_stop_patience))
        if do_fine_tune:
            self.trainer.train()

    @staticmethod
    def compute_metrics(pred):
        """Auxiliary function to add accurate metric into evaluation.
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    def load_dataset(self, data: Dataset, max_length: int = 512):
        print("start mapping")
        r""" Preprocess the input data
        Args:
            dataset (Dataset): the training Dataset generated previously from the pairs of columns
            max_length (int): Maximum length of the input sequence.
        """

        def preprocess_function(examples):
            return self.tokenizer(examples["Col1"], examples["Col2"], max_length=max_length, truncation=True)
        data = data.map(preprocess_function, batched=True)
        return data
