# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import shutil
import time
from logging import getLogger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from learning.MultiAug import MultiCropTableDataset
from learning import model_swav as transformer
from learning.util import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'  # Enable detailed debug information
logger = getLogger()
parser = argparse.ArgumentParser(description="Implementation of SwAV")



#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="datasets/WDC/Test/",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2, 4], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--percentage_crops", type=float, default=[0.5, 0.3], nargs="+",
                    help="crops of tables (example: [0.5, 0.6])")
parser.add_argument("--datasetSize", default=-1, type=int,
                    help="the size of training dataset")

parser.add_argument("--augmentation", type=str, default="sample_cells_TFIDF",
                    help="crops resolutions (example: sample_cells_TFIDF)")
parser.add_argument("--shuffle", default=0.3, type=float,
                    help="portion of views that should be shuffled")
parser.add_argument("--column", dest="column", action="store_true",
                    help="if the unit of input is a column")
parser.add_argument("--header", dest="header", action="store_true",
                    help="if include header in the tables")
parser.add_argument("--subject_column", dest="subject_column", action="store_true",
                    help="if only included subject attributes")
#########################
## swav specific params #
#########################
parser.add_argument("--lm", default="sbert", type=str, help="encoding model")  # arch
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.02, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=5, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=30,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=30, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=5, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=0.4, type=float, help="base learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=0, type=int, help="number of warmup epochs")

parser.add_argument("--dist_url", default="tcp://localhost:12355", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""") #127.0.0.1
parser.add_argument("--world_size", default=1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int, help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=1, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=1,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default="model/SwAV/",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


def main():
    global args
    args = parser.parse_args()
    # args.local_rank = os.environ['LOCAL_RANK']
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    # read the augmentation methods from args
    augmentation_methods = args.augmentation
    if "," in args.augmentation:
        augmentation_methods = args.augmentation.split(",")
    # read the dataset from the data path
    train_dataset = MultiCropTableDataset(
        args.data_path,
        args.nmb_crops,
        args.percentage_crops,
        size_dataset=args.datasetSize,
        shuffle_rate=args.shuffle,
        lm=args.lm,
        subject_column=args.subject_column,
        augmentation_methods=augmentation_methods,
        column=args.column,
        header=args.header)  #

    padder = train_dataset.pad
    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=padder
    )
    logger.info("Building data done with {} tables loaded.".format(len(train_dataset)))

    ### TODO don't know if the device here is appropriate
    device = 'cuda' # if torch.cuda.is_available() else 'cpu'
    # build model
    model = transformer.TransformerModel(
        lm=args.lm,
        device=device,
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes,
        resize=len(train_dataset.tokenizer)
    )
    logger.info("model initialize ...")
    # synchronize batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # copy model to GPU
    model = model.cuda()
    if args.rank == 0:
        logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    optimizer = AdamW(model.parameters(), lr=args.base_lr,weight_decay=args.wd)
    num_steps = (len(train_dataset) // args.batch_size) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)
    logger.info("Building optimizer done.")


    # init mixed precision
    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Initializing mixed precision done.")
    else:
        scaler = None

    # wrap model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_to_work_on],find_unused_parameters=True)  # test distributed

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}

    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=torch.cuda.amp,
        args=args
    )

    start_epoch = to_restore["epoch"]
    print("start_epoch", start_epoch)
    # build the queue
    queue = None
    queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size * args.world_size)
    # cudnn.benchmark = True

    for epoch in range(start_epoch, args.epochs):
        # train the network for one epoch
        logger.info("============ Starting epoch %i ... ============" % epoch)
        # set sampler
        train_loader.sampler.set_epoch(epoch)  # test distibuted
        print(f"Epoch {epoch}")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved() / 1024 ** 2} MB")

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            ###
            #Initialize a new queue with shape (crops_for_assign, queue_length // world_size, feat_dim).
            # len(args.crops_for_assign) indicates the number of crops assigned to the job,
             # args.queue_length // args.world_size indicates the queue length is divided equally among each process
             #  and args.feat_dim indicates the feature dimension. The queue is assigned to the GPU (using .cuda()).
           ###
            queue = torch.zeros(
                len(args.crops_for_assign),  # crops_for_assign
                args.queue_length // args.world_size,
                args.feat_dim,
            ).cuda()
            print("queue shape: ", queue.shape)

        # train the network
        scores, queue = train(train_loader, model, optimizer, epoch, scheduler, scaler, queue)
        training_stats.update(scores)


        # save checkpoints
        if args.rank == 0:
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
            }
            if args.use_fp16:
                # save_dict["amp"] = apex.amp.state_dict()
                save_dict["scaler"] = scaler.state_dict(),  # Save the scaler state
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )
            if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                )
        if queue is not None:
            torch.save({"queue": queue}, queue_path)
        torch.cuda.empty_cache()  # 清理缓存（可选）


def train(train_loader, model, optimizer, epoch, scheduler, scaler, queue):
    def loss_model_fp16(loss):
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def loss_model(loss):
        loss.backward()
        optimizer.step()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    
    use_the_queue = False
    end = time.time()
    for it, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # update learning rate
        iteration = epoch * len(train_loader) + it
        optimizer.zero_grad()

        # normalize the prototypes
        with torch.no_grad():
            w = model.module.prototypes.weight.data.clone()  # for test distributed
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)  # for test distributed

        # ============ multi-res forward passes ... ============
        embedding, output = model(batch)
        #print("embedding", embedding.shape,output.shape)# ,"\n", embedding, "\n", output
        embedding = embedding.detach()
        bs = batch[0].size(0)
        # print("batch size: ", bs)
        # ============ swav loss ... ============
        loss = 0
        for i, crop_id in enumerate(args.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()
                # print("out prototype",out.shape, out)
                # time to use the queue
                if queue is not None:
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
                        out = torch.cat((torch.mm(
                            queue[i],
                            model.module.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                    # print("out embedding for the queue", queue[i, bs:],queue[i, :bs])
                # get assignments
                q = distributed_sinkhorn(out)[-bs:]
                # print("code q is ", q)
            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                x = output[bs * v: bs * (v + 1)] / args.temperature
                # print("Shape of q:", q.shape)
                # print("Shape of x:", x.shape)
                # print("x is ",x)
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(args.nmb_crops) - 1)
        loss /= len(args.crops_for_assign)  # crops_for_assign
        # loss = torch.tensor(loss, device=output.device, dtype=torch.float32)

        # ============ backward and optim step ... ============
        if args.use_fp16:
            with torch.cuda.amp.autocast():
                loss_model_fp16(loss)
        else:
            loss_model(loss)

        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        scheduler.step()
        # ============ misc ... ============
        losses.update(loss.item(), batch[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank == 0 and it % 50 == 0:
            logger.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                .format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    # lr=optimizer.optim.param_groups[0]["lr"],
                )
            )
    return (epoch, losses.avg), queue


@torch.no_grad()
def distributed_sinkhorn(out):
    Q = torch.exp(out / args.epsilon).t()  # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    dist.all_reduce(sum_Q)  # test distributed
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        dist.all_reduce(sum_of_rows)  # test distributed

        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the columns must sum to 1 so that Q is an assignment
    return Q.t()


if __name__ == "__main__":
    main()
# python -m torch.distributed.launch --nproc_per_node=1  E:/Project/CurrentDataset/learning/SwAV.py
