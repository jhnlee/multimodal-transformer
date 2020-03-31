import torch
import os
import logging
from datasets import DataSets

logger = logging.getLogger(__name__)


def get_data(args, dataset, split="train"):
    """ Adapted from original multimodal transformer code """
    alignment = "a" if args.aligned else "na"
    data_path = os.path.join(args.data_path, dataset) + f"_{split}_{alignment}.dt"
    if not os.path.exists(data_path):
        logger.info(f"  - Creating new {split} data")
        data = DataSets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        logger.info(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=""):
    if args.aligned:
        name = name if len(name) > 0 else "aligned_model"
    elif not args.aligned:
        name = name if len(name) > 0 else "nonaligned_model"

    return name + "_" + args.model


def save_model(args, model, name=""):
    name = save_load_name(args, name)
    torch.save(model, f"pre_trained_models/{name}.pt")


def load_model(args, name=""):
    name = save_load_name(args, name)
    model = torch.load(f"pre_trained_models/{name}.pt")
    return model
