# Here, we use sentence-transformers, not just transformers package, so that it is consistent with the paper.
import os
import pickle
import torch.distributed as dist

import torch
from torch.nn.parallel import DataParallel
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader, DistributedSampler
import logging
import math
import random

from Deepjoin.ColToText import ColToTextTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

logger = logging.getLogger(__name__)


def construct_train_dataset(path, naming_file, model_name: str = None, select_num=200,
                            column_to_text_transformation: str = "title-colname-stat-col",
                            shuffle_rate: float = 0.2, seed: int = 42, device="cuda"):
    model = SentenceTransformer(model_name, device=device)  #
    if shuffle_rate > 0:
        shuffle = True
    else:
        shuffle = False
    col_path =os.path.join(path, f'column_representations.pickle')
    if os.path.exists(col_path):
      
      with open(os.path.join(path, f'column_representations.pickle'), 'rb') as f:
        column_representations = pickle.load(f)
      with open(os.path.join(path, f'shuffled_column_representations.pickle'), 'rb') as f:
        shuffled_column_representations = pickle.load(f)
    else:
        col_to_text = ColToTextTransformer(path,  model.tokenizer, naming_file, shuffle=shuffle, select=select_num)
        column_representations, shuffled_column_representations = col_to_text.get_all_column_representations(method=column_to_text_transformation)
        with open(os.path.join(path, f'column_representations.pickle'), 'wb') as f:
          pickle.dump(column_representations, f)
        with open(os.path.join(path, f'shuffled_column_representations.pickle'), 'wb') as f:
          pickle.dump(shuffled_column_representations, f)
    # To get the positive pairs, we follow the approach in OmniMatch, where "positive training pairs are generated
    # based on pairwise cosine similarity of initial column representations (more than or equal to 0.9 ot ensure
    # high true positive rates)".
    flatten_representation = [column_representation
                              for table_name, table_representation in column_representations.items()
                              for column_name, column_representation in table_representation.items()]
    print(len(flatten_representation))

    embeddings = model.encode(flatten_representation, device=device)
    similarities = cos_sim(embeddings, embeddings)
    positive_pairs = set()
    positive_pairs_indices = set()
    for i in range(len(similarities) - 1):
        for j in range(i + 1, len(similarities)):
            sim = similarities[i][j].item()
            if sim >= 0.9 and (i, j) not in positive_pairs_indices and (j, i) not in positive_pairs_indices:
                positive_pairs_indices.add((i, j))
                positive_pairs.add((flatten_representation[i], flatten_representation[j]))  # (X, Y)
    if shuffle_rate > 0:
        # sample from the positive pairs
        random.seed(seed)
        to_be_shuffled = random.sample(list(positive_pairs_indices), int(shuffle_rate * len(positive_pairs_indices)))
        shuffled_flatten_representation = [column_representation for table_name, table_representation
                                           in shuffled_column_representations.items()
                                           for column_name, column_representation in table_representation.items()]
        for i, j in to_be_shuffled:
            positive_pairs.add((shuffled_flatten_representation[i], flatten_representation[j]))  # (X', Y)
    print(len(positive_pairs))
    return list(positive_pairs)


def train_model(model_name: str, train_dataset, dev_samples=None, model_save_path: str = None, batch_size: int = 32,
                 warmup_steps: int = None, weight_decay: float = 0.01, num_epochs: int = 3,
                device="cuda", cpuid=2):#learning_rate: float = 2e-5,dist_url=None,,  world_size=1
    # Here we define our SentenceTransformer model
    #### Just some code to print debug information to stdout
    #rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    #dist.init_process_group(backend='gloo', init_method=dist_url, world_size=world_size, rank=rank)

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    model = SentenceTransformer(model_name)
    model = model.to(device)
    

    if device == "cuda":
        # set cuda
        if cpuid == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        elif cpuid == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        elif cpuid == 2:
            device_ids = [0, 1]
            torch.cuda.set_device(device_ids[0])
            model = DataParallel(model, device_ids=[0, 1])
            model = model.module
        else:
            pass

    # Special data loader that avoid duplicates within a batch
    train_dataset = [InputExample(texts=[a, b], label=1) for a, b in train_dataset]
    sentence_label_dataset = SentenceLabelDataset(train_dataset)
    #train_sampler = DistributedSampler(sentence_label_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(sentence_label_dataset, batch_size=batch_size)#, sampler=train_sampler
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    # Configure the training
    if warmup_steps is None:
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logging.info(f"Warmup-steps: {warmup_steps}, "
                 f"Weight-decay: {weight_decay}, "
                 f"Batch-size: {batch_size}, "
                 f"Num-epochs: {num_epochs}")
    if dev_samples is not None:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples)
    else:
        evaluator = None
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        evaluator=evaluator,
        #optimizer_params={"lr": learning_rate},
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        output_path=model_save_path,
        use_amp=True,  # Set to True, if your GPU supports FP16 operations
    )
    logging.info("Training completed.")
    
    return model
