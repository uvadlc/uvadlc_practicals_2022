################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

"""Main driver script to run the code."""
import os
import argparse
import torch
from learner import Learner


def parse_option():
    parser = argparse.ArgumentParser("Visual Prompting for CLIP")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num of workers to use"
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="number of training epochs"
    )
    parser.add_argument(
        "--square_size",
        type=int,
        default=8,
        help="size of each square in checkboard prompt",
    )
    # optimization
    parser.add_argument("--optim", type=str, default="sgd", help="optimizer to use")
    parser.add_argument("--learning_rate", type=float, default=40, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument(
        "--warmup", type=int, default=1000, help="number of steps to warmup for"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--patience", type=int, default=1000)

    # model
    parser.add_argument("--model", type=str, default="clip")
    parser.add_argument("--arch", type=str, default="ViT-B/32")
    parser.add_argument(
        "--method",
        type=str,
        default="padding",
        choices=[
            "padding",
            "random_patch",
            "fixed_patch",
        ],
        help="choose visual prompting method",
    )
    parser.add_argument(
        "--prompt_size", type=int, default=30, help="size for visual prompts"
    )
    parser.add_argument(
        "--text_prompt_template", type=str, default="This is a photo of a {}",
    )
    parser.add_argument(
        "--visualize_prompt",
        action="store_true",
        help="visualize the (randomly initialized) prompt and save it to a file for debugging",
    )

    # dataset
    parser.add_argument("--root", type=str, default="./data", help="dataset")
    parser.add_argument("--dataset", type=str, default="cifar100", help="dataset")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument(
        "--test_noise", default=False, action="store_true",
        help="whether to add noise to the test images",
    )

    # other
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for initializing training"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./save/models", help="path to save models"
    )
    parser.add_argument(
        "--image_dir", type=str, default="./save/images", help="path to save images"
    )
    parser.add_argument("--filename", type=str, default=None, help="filename to save")
    parser.add_argument("--trial", type=int, default=1, help="number of trials")
    parser.add_argument(
        "--resume", type=str, default=None, help="path to resume from checkpoint"
    )
    parser.add_argument(
        "--evaluate", default=False, action="store_true", help="evaluate model test set"
    )
    parser.add_argument("--gpu", type=int, default=None, help="gpu to use")
    parser.add_argument(
        "--use_wandb", default=False, action="store_true", help="whether to use wandb"
    )

    args = parser.parse_args()

    args.num_workers = min(args.num_workers, os.cpu_count())

    args.filename = "{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}".format(
        args.method,
        args.prompt_size,
        args.dataset,
        args.model,
        args.arch,
        args.optim,
        args.learning_rate,
        args.weight_decay,
        args.batch_size,
        args.warmup,
        args.trial,
    )

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    return args


def main():
    args = parse_option()
    print(args)
    learn = Learner(args)

    if args.evaluate:
        learn.evaluate("test")
    else:
        learn.run()
        learn.evaluate("valid")
        learn.evaluate("test")


if __name__ == "__main__":
    main()
