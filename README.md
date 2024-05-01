# SILM


### Training Requirements

* Python 3.11.0
* PyTorch 2.2.1
* Transformers 4.38.2
* NVIDIA Apex

Install requirements:
```
pip install -r requirements
```

### Datasets
All datasets, including GoogleSearch(for testing running time), WDC: 602 tables from Web Data Commons
and GDS: 660 selected tables from Google Dataset Search, can be downloaded via this link
https://drive.google.com/file/d/1nW2zvw_m9RL_GZQetSQYC6Sp_WyPt0ef/view?usp=drive_link.

### Running the offline pre-training pipeline:


The main entry point is `pretrain_all.py`.
If you only want to train using subject attributes, please write `--subject_column \` in the script.
Script for fine-tuned training and encoding is `train.sh`, script for pretrain language models encoding
is `pretrain.sh`.
Script for fine-tuning training and encoding using different augmentation times is `Aug.sh`.


Hyperparameters:

* `--batch_size`, `--lr`, `--n_epochs`, `--max_len`: standard batch size, learning rate, number of training epochs, max sequence length
* `--lm`: the language model (we use roberta for all the experiments)
* `--size`: the maximum number of tables/columns used during pre-training
* `--projector`: the dimension of projector
* `--save_model`: if this flag is on, the model checkpoint will be saved to the directory specified in the `--logdir` flag
* `--augment_op`: augmentation operator for contrastive learning. Using "sample cells TFIDF", uses sample_cells_TFIDF; Using "sample cells", using sample_cells
* `--fp16`: half-precision training (always turn this on)
* `--save_model`: whether to save the vectors in a pickle file, which is then used in the online processing


### P1-P4

The main entry point is  `main.py`.   Scripts:.
* Script for Phase 1 type inference is `P1.sh`, the baseline script is `runBaselineP1.sh`.
* Script for Phase 2 conceptual attribute inference and Phase 3 hierarchy inference is `P23.sh`, 
the baseline script is 'runBaselineP23.sh'.
* Script for Phase 4 conceptual attribute relationship search is `P4.sh`.
Hyperparameters
* `--dataset`: dataset name 
* `--embed`: name of used language name, only support sbert, bert. roberta
* `--baseline`: if this flag is on, we will run D3L as our baseline
* `--clustering`: the clustering algorithm name, defined as agglomerative clustering

* `--iteration`: iteration over clustering.
* `--subjectCol`: if we only take the subject attributes' embedding while using all the embeddings of all tables' attributes.

* `--step`: different phases, default 1 as Phase 1
* `--estimateNumber`: the estimate cluster number

* `--intervalSlice`: the interval when slicing the dendrogram
* `--delta`: delta for best silhouette score in Phase 3 -- hierarchy inference

* `--similarity`: the similarity threshold in conceptual attribute relationship search
* `--portion`: the fraction threshold for non-subject attributes in relationship search
* `--Euclidean`: distance metric, if this flag is on, we will use Euclidean distance


* `--SelectType`: This is for end-to-end part, input is the top level types in the ground truth, if this is "", we default
this as all inputting all tables
* `--P1Embed`: the embedding file for P1 
* `--P23Embed`: the embedding file for P2 & P3 
* `--P4Embed`: the embedding file for P4


### End to end evaluation


### Running time
The main entry point is `main.py`. Script for training is `Runtime.sh`.
* `--tableNumber`: the number of tables when testing the running time.