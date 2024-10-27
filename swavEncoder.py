import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm  # Ensure correct import

from Utils import subjectCol, mkdir
from clustering import data_classes, evaluate_cluster
from UnicornTest import find_clusters
from learning.util import restart_from_checkpoint

from learning import model_swav as transformer
from learning.MultiAug import MultiCropTableDataset


def Encode(model: transformer.TransformerModel,
           unlabeled: MultiCropTableDataset,
           batch_size=128,
           total=None,cls=True):
    tables_dict = unlabeled.data()
    tables = list(tables_dict.values())
    total = total if total is not None else len(tables)
    batch = []
    results = []
    for tid, table in tqdm(enumerate(tables), total=total):
        x, _ = unlabeled._tokens(table)
        batch.append((x, x, []))
        if tid == total - 1 or len(batch) == batch_size:
            # model inference
            with torch.no_grad():
                x, _, _ = unlabeled.pad(batch)
                # print("All", x,len(x))
                # all column vectors in the batch
                column_vectors = model.infer(x)

                if cls is False:
                    list_of_tensors = column_vectors.cpu().numpy()
                    for tensor in list_of_tensors:
                        results.append(tensor[np.newaxis, :])
                    #print(column_vectors.tolist()[0], "\n", column_vectors.shape)
                # print("column_vectors ", column_vectors,column_vectors.shape)
                else:
                    ptr = 0
                    for xi in x:
                        current = []
                        for token_id in xi:
                            if token_id == unlabeled.tokenizer.cls_token_id:
                                current.append(column_vectors[ptr].cpu().numpy())
                                ptr += 1
                        results.append(current)
            batch.clear()
    return unlabeled.samples, results


def remove_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def encoding(dataPath, isTransfer="",cls=True):
    checkpoint = torch.load(os.path.join(dataPath, "checkpoint.pth.tar"), map_location=torch.device('cuda'))
    state_dict_model = checkpoint['state_dict']
    state_dict_model = remove_module_prefix(state_dict_model)
    args = checkpoint['args']
    print(args)
    if 'transformer.embeddings.position_ids' in state_dict_model:
        del state_dict_model['transformer.embeddings.position_ids']
    if isTransfer != "":
        args.data_path = isTransfer
    train_dataset = MultiCropTableDataset(
        args.data_path,
        args.nmb_crops,
        args.percentage_crops,
        shuffle_rate=args.shuffle,
        augmentation_methods=args.augmentation,
        subject_column= args.subject_column,
        header=args.header,
        column=args.column,
        lm=args.lm)  #
    print(train_dataset[0])

    model = transformer.TransformerModel(
        lm=args.lm,
        device="cuda",
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,#
        nmb_prototypes=args.nmb_prototypes,
        resize=len(train_dataset.tokenizer),
        cls=cls
    )
    model = model.to("cuda")

    model.load_state_dict(state_dict_model)
    tables, results = Encode(model, train_dataset, batch_size=32,cls=cls)
    dfs_count = 0
    dataEmbeds = []

    for i, file in enumerate(tables):
        if args.column is True:
            file = file.replace(".csv|", ".")
        dfs_count += 1
        cl_features_file = np.array(results[i])
        dataEmbeds.append((file, cl_features_file))
    print( dataEmbeds[0],dataEmbeds[0][1].shape)
    return dataEmbeds

"""
The following is a simple example of how to use the generated fine-tuned model to encode dataset.
"""
dataset = "WDC"
output_path_embedding = f"model/SwAV/{dataset}/42HTT/" #
encode_path = f"datasets/{dataset}/Test/"#
embeddings = encoding(output_path_embedding,encode_path,cls=False)
out = f"result/embedding/{dataset}/"
mkdir(out)
embedding_name = f"{dataset}_sbert_42HTT_column.pkl"
with open(os.path.join(out,embedding_name), "wb") as f:
    pickle.dump(embeddings, f)
