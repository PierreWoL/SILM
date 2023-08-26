import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

import TableAnnotation as TA
from SubjectColumnDetection import ColumnType

lm_mp = {'roberta': 'roberta-base',
         'bert': 'bert-base-uncased',
         'distilbert': 'distilbert-base-uncased',
         'sbert': 'sentence-transformers/all-mpnet-base-v2'}


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")


def are_all_numbers(values):
    cleaned_values = [val for val in values if
                      pd.notna(val) and str(val).strip() != '' and val not in ["n / a", "n/a", "N/A"]]
    return all(val.isdigit() for val in cleaned_values)


def aug(table: pd.DataFrame):
    exists = []
    for index in range(0, table.shape[1]):
        if are_all_numbers(table.iloc[:, index][0].split(",")) is False:
            exists.append(index)
    return table.iloc[:, exists]


def subjectCol(table: pd.DataFrame, combine=False):
    sub_cols_header = []
    anno = TA.TableColumnAnnotation(table, isCombine=combine)
    types = anno.annotation
    for key, type in types.items():
        if type == ColumnType.named_entity:
            sub_cols_header = [table.columns[key]]
            break
    return sub_cols_header


"""import pickle
file_path = "/mnt/d/CurrentDataset/result/embedding/starmie/vectors/TabFact/cl_drop_num_col_lm_sbert_head_column_0_header.pkl"
with open(file_path, "rb") as file:
    G = pickle.load(file)
print(G)"""

# Load the RoBERTa tokenizer and model
"""tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Sentence to encode
special_tokens_dict = {
    'additional_special_tokens': ["<subjectcol>", "<header>", "</subjectcol>", "</header>"]}
header_token = ('<header>', '</header>')
SC_token = ('<subjectcol>', '</subjectcol>')
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.special_tokens_map.items()

input_sentence = "<s> <header> 0 </header> Greater Atlantic City Chamber Board of Directors Meeting Ambassador/Membership Committee Public Policy Leadership Series: Steve Callender," \
                 " President the Casino Association New Jersey 2020 Golf Classic presented by ShopRite Executive ‘100 Faces War’ Featured in Smithsonian Traveling Exhibition at Noyes Arts Garage Events Captain's Table Reception Progressive Insurance Boat Show "
# Update the tokenizer's vocabulary size

model_config = AutoConfig.from_pretrained("roberta-base")
tokenizer_vocab_size = tokenizer.vocab_size + len(special_tokens_dict['additional_special_tokens'])
model_config.vocab_size = tokenizer_vocab_size
model = AutoModel.from_pretrained("roberta-base")
model.resize_token_embeddings(new_num_tokens=tokenizer_vocab_size)
# Tokenize the sentence
tokens = tokenizer.encode_plus(input_sentence, add_special_tokens=True, max_length=512,
                               truncation=True, return_tensors="pt")
# Perform the encoding using the model
with torch.no_grad():
    outputs = model(**tokens)

# Extract the last hidden state (embedding) from the outputs
last_hidden_state = outputs.last_hidden_state

# Depending on your application, you might want to use a pooling strategy to get a fixed-size sentence embedding.
# Here, we'll use mean pooling as in your original code snippet.
pooled_embedding = last_hidden_state.mean(dim=1)
"""