"""
This module is the training part for the contrastive learning.
Based on https://github.com/megagonlabs/starmie/
"""
import torch
import pandas as pd
import os
from Utils import simplify_string, subjectCol
from .model import BarlowTwinsSimCLR
from pretrainData import PretrainTableDataset
from tqdm import tqdm
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import List


def train_step(train_iter, model, optimizer, scheduler, scaler, hp):
    """
    Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (BarlowTwinsSimCLR): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        scaler (GradScaler): gradient scaler for fp16 training
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """

    def loss_model_fp16(loss):
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def loss_model(loss):
        loss.backward()
        optimizer.step()

    num_augs = 2  # default value
    if "," in hp.augment_op:
        num_augs = len(hp.augment_op.split(','))

    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        if num_augs <= 2:
            x_ori, x_aug, cls_indices = batch
            #print(cls_indices)
            losses = [model(x_ori, x_aug, cls_indices, mode='simclr')]
        else:
            x_vals = batch[:-1]  # Separate out cls_indices
            cls_indices = batch[-1]
            #print(cls_indices)
            # Compute all unique combinations of x_vals and calculate loss
            losses = []
            for j in range(len(x_vals)):
               for k in range(j + 1, len(x_vals)):
                   cls_indices_ij = (cls_indices[j],cls_indices[k])
                   #print(cls_indices_ij)
                   loss = model(x_vals[j], x_vals[k],cls_indices_ij, mode='simclr')
                   losses.append(loss)
        #avg_loss = sum(losses) / len(losses)
        # 使用 torch.stack 将这些张量堆叠成一个新的张量
        stacked_tensors = torch.stack(losses)

        # 计算平均值
        avg_loss = torch.mean(stacked_tensors)
        if hp.fp16:
            with torch.cuda.amp.autocast():
                loss_model_fp16(avg_loss)
        else:
            loss_model(avg_loss)

        scheduler.step()

        if i % 10 == 0:  # monitoring
            print(f"step: {i}, losses: {tuple(l.item() for l in losses)}")

        for loss in losses:
            del loss



def train(trainset, hp):
    """Train and evaluate the model

    Args:
        trainset (PretrainTableDataset): the training set
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)
    Returns:
        The pre-trained table model
    """
    padder = trainset.pad

    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm, resize=len(trainset.tokenizer)) #
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)
    if hp.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    for epoch in range(1, hp.n_epochs + 1):
        print("epoch is ", epoch)
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, scaler, hp)
        
        # save the last checkpoint
        if hp.save_model and epoch == hp.n_epochs:
            directory = os.path.join(hp.logdir, hp.dataset)
            op_augment_new =simplify_string(hp.augment_op)
            if not os.path.exists(directory):
                os.makedirs(directory)
                

            # save the checkpoints for each component
            ckpt_path = os.path.join(hp.logdir,  hp.dataset, 'model_' +
                                     op_augment_new + "_lm_" + str(
                hp.lm) + "_" + str(hp.sample_meth) + "_" + str(
                hp.table_order) + '_' + str(
                hp.run_id) + "_" + str(hp.check_subject_Column) + ".pt")
            if hp.single_column:
                ckpt_path = os.path.join(hp.logdir, hp.dataset, 'model_' +
                                        op_augment_new + "_lm_" + str(
                    hp.lm) + "_" + str(hp.sample_meth) + "_" + str(
                    hp.table_order) + '_' + str(
                    hp.run_id) + "_" + str(hp.check_subject_Column) + "_singleCol.pt")
            elif hp.subject_column:
                ckpt_path = os.path.join(hp.logdir,  hp.dataset, 'model_' +
                                        op_augment_new + "_lm_" + str(
                    hp.lm) + "_" + str(hp.sample_meth) + "_" + str(
                    hp.table_order) + '_' + str(
                    hp.run_id) + "_" + str(hp.check_subject_Column) + "_subCol.pt")
                print(ckpt_path)
            elif hp.header:
                ckpt_path = os.path.join(hp.logdir,  hp.dataset, 'model_' +
                                        op_augment_new + "_lm_" + str(
                    hp.lm) + "_" + str(hp.sample_meth) + "_" + str(
                    hp.table_order) + '_' + str(
                    hp.run_id) + "_" + str(hp.check_subject_Column) + "_header.pt")
            elif hp.column:
                ckpt_path = os.path.join(hp.logdir,  hp.dataset, 'model_' +
                                         op_augment_new + "_lm_" + str(
                    hp.lm) + "_" + str(hp.sample_meth) + "_" + str(
                    hp.table_order) + '_' + str(
                    hp.run_id) + "_" + str(hp.check_subject_Column) + "_column.pt")
            ckpt = {'model': model.state_dict(),
                    'hp': hp}
            torch.save(ckpt, ckpt_path)
            print("Save models! to the path", ckpt_path)

            # test loading checkpoints
            # load_checkpoint(ckpt_path)
        # intrinsic evaluation with column matching


def inference_on_tables(tables: List[pd.DataFrame],
                        model: BarlowTwinsSimCLR,
                        unlabeled: PretrainTableDataset,
                        batch_size=128,
                        total=None,
                        subject_column=False):
    """Extract column vectors from a table.

    Args:
        tables (List of DataFrame): the list of tables
        model (BarlowTwinsSimCLR): the model to be evaluated
        unlabeled (PretrainTableDataset): the unlabeled dataset
        batch_size (optional): batch size for model inference

    Returns:
        List of np.array: the column vectors
    """
    total = total if total is not None else len(tables)
    batch = []
    results = []
    for tid, table in tqdm(enumerate(tables), total=total):
        # print(tid, table)
        if subject_column is True:
            cols = subjectCol(table)
            if len(cols) > 0:
                    table = table[cols]
        x, _ = unlabeled._tokenize(table)
        batch.append((x, x, []))
        if tid == total - 1 or len(batch) == batch_size:
            # model inference
            with torch.no_grad():
                x, _, _ = unlabeled.pad(batch)
                # print("All",x, _, _)
                # all column vectors in the batch
                column_vectors = model.inference(x)
                # print("column_vectors ", column_vectors)
                ptr = 0
                for xi in x:
                    current = []
                    for token_id in xi:
                        if token_id == unlabeled.tokenizer.cls_token_id:
                            current.append(column_vectors[ptr].cpu().numpy())
                            ptr += 1
                    results.append(current)
            batch.clear()

    return results


def load_checkpoint(ckpt):
    """Load a model from a checkpoint.
        ** If you would like to run your own benchmark, update the ds_path here
    Args:
        ckpt (str): the model checkpoint.

    Returns:
        BarlowTwinsSimCLR: the pre-trained model
        PretrainDataset: the dataset for pre-training the model
    """
    hp = ckpt['hp']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(device)
    # todo change the resize by passing a parameter
    Resize = -1
    if hp.lm == 'roberta':
        Resize = 50269
    elif hp.lm == 'sbert':
        Resize = 30529
    elif hp.lm == 'bert':
        Resize = 30526
  
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm,resize=Resize)  # 50269 30529  
    model = model.to(device)
    model.load_state_dict(ckpt['model'])
    ds_path = os.path.join(os.getcwd(), 'datasets/%s/Test' % hp.dataset)
    dataset = PretrainTableDataset.from_hp(ds_path, hp)
    return model, dataset
