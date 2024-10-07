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

"""
print("PyTorch Version: ", torch.__version__)
print("CUDA Available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Current CUDA Device: ", torch.cuda.current_device())
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Please check your installation.")


# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device available: ", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA device not available, using CPU")

# 创建一个张量并将其移动到GPU
x = torch.randn(3, 3)
x = x.to(device)
print("Tensor x on device:", x.device)

# 在GPU上进行张量计算
y = torch.randn(3, 3).to(device)
z = x + y
print("Result of x + y on GPU:", z)

# 将结果移动回CPU并打印
z = z.to("cpu")
print("Result moved back to CPU:", z)


tensor1 = torch.randn( 3,4)
tensor2 = torch.randn(3,4)
tensor3 = torch.randn(3,4)
print(tensor1,"\n", tensor2,"\n",  tensor3)
# 在第0维拼接，即增加行数
concat0 = torch.cat((tensor1, tensor2, tensor3), dim=0)
print("Concatenated along dimension 0:")
print(concat0)
print("Shape of concat0:", concat0.shape)  # 应该是 (9, 4)



# 示例张量
b1 = torch.tensor([
    [1,  2,  3],
    [21,  22,  23],
    [19,  20,  21]
])
b2 = torch.tensor([
    [21,  22,  23],
    [221,  222,  223],
    [129,  220,  221]
])
b3 = torch.tensor([
    [251,  252,  253],
    [2251,  2252,  2523],
    [1259,  2520,  2521]
])

bn = [b1,b2,b3]
c = [torch.empty(0) for i in range(len(b1))]
for bi in bn:
    for index in range(bi.size(0)):
        c[index] = torch.cat((c[index], bi[index].unsqueeze(0)), dim=0)
print(c)
emebddings=torch.cat(c, dim=0)
print(emebddings)

c2 = [torch.cat([b[i].unsqueeze(0) for b in bn], dim=0) for i in range(b1.size(0))]
"""

"""dataset = "GDS"
with open(os.path.join(f"datasets/{dataset}/", f'UnicornP1Result{dataset}Zero.pickle'), 'rb') as f:
    data_pairs = pickle.load(f)
print("load successfully")
data_path = os.getcwd() + f"/datasets/{dataset}/Test/"
table_names = [i for i in os.listdir(data_path) if i.endswith(".csv")]
ground_truth = os.getcwd() + f"/datasets/{dataset}/groundTruth.csv"
clusters = find_clusters(data_pairs,table_names)
total = 0
for index, cluster in clusters.items():
     total+=len(cluster)
gt_clusters, ground_t, gt_cluster_dict = data_classes(data_path, ground_truth)
gt_clusters0, ground_t0, gt_cluster_dict0 = data_classes(data_path, ground_truth, superclass=False)
del ground_t0, gt_cluster_dict0
folderName = os.getcwd() + f"/datasets/{dataset}"
metrics_value = evaluate_cluster(gt_clusters, gt_cluster_dict, clusters, None,
                                 gt_clusters0)
print(metrics_value)"""

"""
data_names = {i:i.split("_")[0] for i in os.listdir("D:\datasets\Backups\WebData") if i.endswith(".csv")}
df = pd.DataFrame(list(data_names.items()), columns=['table', 'name'])
df.to_csv("D:\datasets\Backups\WebData\\naming.csv")
print(df)"""
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

dataset = "WDC"
path = f"model/SwAV/{dataset}/42HTT/" #
encode_path = f"datasets/{dataset}/Test/"#
embeddings = encoding(path,encode_path,cls=False)
out = f"result/embedding/{dataset}/"
mkdir(out)
with open(os.path.join(out, f"{dataset}_sbert_42HTT_column.pkl"), "wb") as f:
    pickle.dump(embeddings, f)
