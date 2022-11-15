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

"""Helper script to evaluate cross-dataset robustness."""
import os
import argparse
import torch
from learner import Learner
from clip import clip
from dataset import load_dataset, construct_dataloader
from pprint import pprint
from utils import DummyArgs


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
        "--text_prompt_template",
        type=str,
        default="This a photo of a {}",
    )

    # dataset
    parser.add_argument("--root", type=str, default="./data", help="dataset")
    parser.add_argument("--dataset", type=str, default="cifar100", help="dataset")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument(
        "--test_noise",
        default=False,
        action="store_true",
        help="whether to add noise to the test images",
    )

    parser.add_argument(
        "--visualize_prompt",
        action="store_true",
        help="visualize the (randomly initialized) prompt and save it to a file for debugging",
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
    # 1. Load the pre-trained model, trained on either CIFAR10 or CIFAR100
    args = parse_option()
    print(args)

    assert (
        args.resume
    ), "Set argument --resume to set the path to the best saved model checkpoint"

    learn = Learner(args)

    if args.evaluate:

        # Load clip image transformation
        _, preprocess = clip.load(args.arch)

        # 2. Load the test datset from CIFAR10
        dummy_args = DummyArgs()
        dummy_args.test_noise = args.test_noise
        dummy_args.root = args.root
        dummy_args.dataset = "cifar10"
        _, _, cifar10_test = load_dataset(dummy_args, preprocess)

        # 3. Load the test dataset from CIFAR100
        dummy_args = DummyArgs()
        dummy_args.test_noise = args.test_noise
        dummy_args.root = args.root
        dummy_args.dataset = "cifar100"
        _, _, cifar100_test = load_dataset(dummy_args, preprocess)

        # 4. Combine the classnames from CIFAR10 and CIFAR100.
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Define `classnames` as a list of 10 + 100 class labels from CIFAR10 and CIFAR100

        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

        classnames = cifar10_test.classes + cifar100_test.classes

        # 5. Load the clip model
        print(f"Loading CLIP (backbone: {args.arch})")
        clip_model = learn.vpt.load_clip_to_cpu(args)
        clip_model.to(args.device)
        # Hack to make model as float() (This is a CLIP hack)
        if args.device == "cpu":
            clip_model = clip_model.float()

        # 6. Construct text prompts
        template = args.text_prompt_template
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        print("List of prompts:")
        pprint(prompts)

        # 7. Compute the text features
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Compute the text features (for each of the prompts defined above) using CLIP
        # Note: This is similar to the code you wrote in `clipzs.py`

        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

        # 8. Set the text_features of pre-trained model to the calculated text features
        learn.vpt.text_features = text_features

        # 9. Offset the target labels of CIFAR100 by 10
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Add an offset of 10 to the targets of CIFAR100
        # That is, if a class in CIFAR100 corresponded to '4', it should now correspond to '14'
        # Set the result of this to the attribute cifar100_test.targets to override them

        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

        # 10. Define the dataloader for CIFAR10
        cifar10_loader = construct_dataloader(args, cifar10_test)

        # 11. Define the dataloader for CIFAR100
        cifar100_loader = construct_dataloader(args, cifar100_test)

        # 12. Update the test loader of learner to CIFAR10
        learn.test_loader = cifar10_loader

        # 13. Evaluate the performance of the model
        acc_cifar10 = learn.evaluate("test")

        # 14. Update the test loader of learner to CIFAR100
        learn.test_loader = cifar100_loader

        # 15. Evaluate the performance of the model
        acc_cifar100 = learn.evaluate("test")

        # 16. Compute the weighted average (or total performance)
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Compute the weighted average of the above two accuracies

        # Hint:
        # - accurary_all = acc_cifar10 * (% of cifar10 samples) \
        #                  + acc_cifar100 * (% of cifar100 samples)

        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

        print(f"TOP1 Accuracy on cifra10 + cifar100 is: {accuracy_all}")
        exit()
    else:
        raise ValueError("Enable flag --evaluate!")


if __name__ == "__main__":
    main()
