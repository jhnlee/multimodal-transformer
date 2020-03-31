import argparse
import logging
import random
import os
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets import DataSets
from model import MULTModel
from utils import *
from eval_metrics import eval_iemocap
from warmup_scheduler import GradualWarmupScheduler

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset, model):
    sampler = (
        RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    )
    args.batch_size = args.batch_size * max(1, args.n_gpu)
    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=args.batch_size,)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    t_total = len(train_loader) * args.num_epochs
    args.warmup_step = int(args.warmup_percent * t_total)

    if args.warmup_step != 0:
        scheduler_plateau = ReduceLROnPlateau(
            optimizer, "min", patience=args.when, factor=0.1, verbose=True
        )
        scheduler = GradualWarmupScheduler(
            optimizer, 1, args.warmup_step, after_scheduler=scheduler_plateau
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=args.when, factor=0.1, verbose=True
        )
    loss_fct = torch.nn.CrossEntropyLoss()

    # Train!
    logger.info("***** Running Multimodal Transformer *****")
    logger.info("  Num Epochs = %d", args.num_epochs)
    logger.info(
        "  Total train batch size = %d",
        args.batch_size * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup Steps = %d", args.warmup_step)

    global_step = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_val_loss = 1e8

    model.zero_grad()
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    train_iterator = trange(
        0, int(args.num_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_loader, desc="Iteration", disable=args.local_rank not in [-1, 0],
        )
        for step, (batch_x, batch_y, meta) in enumerate(epoch_iterator):
            model.train()
            sample_idx, text, audio, vision = list(map(lambda x: x.to(args.device), batch_x))
            batch_y = batch_y.squeeze(-1).long().to(args.device)
            preds, hidden = model(vision, audio, text)
            loss = loss_fct(preds.view(-1, 2), batch_y.view(-1))
            if args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and global_step % args.logging_steps == 0:
                logger.info("  train loss : %.3f", (tr_loss - logging_loss) / args.logging_steps)
                logging_loss = tr_loss

        if args.local_rank in [-1, 0]:
            val_loss, preds, labels = evaluate(args, eval_dataset, model, loss_fct)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info("  Saved model")
                save_model(args, model, name=args.name)

            logger.info("  val loss : %.3f", val_loss)
            logger.info("  best_val loss : %.3f", best_val_loss)


def evaluate(args, eval_dataset, model, loss_fct):

    model.eval()
    sampler = SequentialSampler(eval_dataset)
    eval_loader = DataLoader(eval_dataset, sampler=sampler, batch_size=args.batch_size,)
    total_loss = 0.0
    nb_eval_steps = 0

    total_preds = []
    total_labels = []

    for batch_x, batch_y, meta in tqdm(eval_loader, desc="Evaluating"):
        with torch.no_grad():
            sample_idx, text, audio, vision = list(map(lambda x: x.to(args.device), batch_x))
            batch_y = batch_y.squeeze(-1).long().to(args.device)
            preds, hidden = model(vision, audio, text)
            loss = loss_fct(preds.view(-1, 2), batch_y.view(-1))

        total_loss += loss.mean().item()
        total_preds.append(preds.view(-1, 2))
        total_labels.append(batch_y.view(-1))
        nb_eval_steps += 1
    loss = total_loss / nb_eval_steps

    return loss, torch.cat(total_preds), torch.cat(total_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="MulT", help="name of the model to use (Transformer, etc.)"
    )

    # Tasks
    parser.add_argument(
        "--do_vision",
        action="store_true",
        help="use the crossmodal fusion into vision (default: False)",
    )
    parser.add_argument(
        "--do_audio",
        action="store_true",
        help="use the crossmodal fusion into audio (default: False)",
    )
    parser.add_argument(
        "--do_text",
        action="store_true",
        help="use the crossmodal fusion into text (default: False)",
    )
    parser.add_argument(
        "--aligned", action="store_true", help="consider aligned experiment or not (default: False)"
    )
    parser.add_argument(
        "--dataset", type=str, default="iemocap", help="dataset to use (default: iemocap)"
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="path for storing the dataset"
    )
    parser.add_argument(
        "--save_path", type=str, default="pre_trained_models", help="path for storing the dataset"
    )

    # Dropouts
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
    parser.add_argument(
        "--attn_dropout_a", type=float, default=0.0, help="attention dropout (for audio)"
    )
    parser.add_argument(
        "--attn_dropout_v", type=float, default=0.0, help="attention dropout (for visual)"
    )
    parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
    parser.add_argument("--embed_dropout", type=float, default=0.25, help="embedding dropout")
    parser.add_argument("--res_dropout", type=float, default=0.1, help="residual block dropout")
    parser.add_argument("--out_dropout", type=float, default=0.0, help="output layer dropout")

    # Architecture
    parser.add_argument(
        "--layers", type=int, default=4, help="number of layers in the network (default: 5)"
    )
    parser.add_argument(
        "--d_model", type=int, default=30, help="dimension of layers in the network (default: 30)"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=10,
        help="number of heads for the transformer network (default: 5)",
    )
    parser.add_argument(
        "--attn_mask",
        action="store_false",
        help="use attention mask for Transformer (default: true)",
    )

    # Tuning
    parser.add_argument(
        "--batch_size", type=int, default=45, metavar="N", help="batch size (default: 24)"
    )
    parser.add_argument(
        "--clip", type=float, default=0.8, help="gradient clip value (default: 0.8)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-3, help="initial learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--optim", type=str, default="Adam", help="optimizer to use (default: Adam)"
    )
    parser.add_argument("--num_epochs", type=int, default=40, help="number of epochs (default: 40)")
    parser.add_argument(
        "--when", type=int, default=10, help="when to decay learning rate (default: 20)"
    )
    parser.add_argument(
        "--batch_chunk", type=int, default=1, help="number of chunks per batch (default: 1)"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training")
    parser.add_argument(
        "--warmup_percent", default=0.1, type=float, help="Linear warmup over warmup_percent."
    )

    # Logistics
    parser.add_argument(
        "--logging_steps", type=int, default=30, help="frequency of result logging (default: 30)"
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
    parser.add_argument(
        "--name", type=str, default="mult", help='name of the trial (default: "mult")'
    )
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    set_seed(args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    train_data = get_data(args, args.dataset, "train")
    eval_data = get_data(args, args.dataset, "valid")
    test_data = get_data(args, args.dataset, "test")

    orig_d_t, orig_d_a, orig_d_v = train_data.get_dim()

    model = MULTModel(
        only_vision=args.do_vision,
        only_audio=args.do_audio,
        only_text=args.do_text,
        orig_d_v=orig_d_v,
        orig_d_a=orig_d_a,
        orig_d_t=orig_d_t,
        n_head=args.num_heads,
        n_cmlayer=args.layers,
        d_model=args.d_model,
        emb_dropout=args.embed_dropout,
        attn_dropout=args.attn_dropout,
        attn_dropout_audio=args.attn_dropout_a,
        attn_dropout_vision=args.attn_dropout_v,
        relu_dropout=args.relu_dropout,
        res_dropout=args.res_dropout,
        out_dropout=args.out_dropout,
        max_position=128,
        attn_mask=args.attn_mask,
        scale_embedding=True,
    ).to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train(args, train_data, eval_data, model)
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        model = load_model(args, name=args.name).to(args.device)
        _, preds, labels = evaluate(args, test_data, model, torch.nn.CrossEntropyLoss())
        eval_iemocap(preds, labels)


if __name__ == "__main__":
    main()
