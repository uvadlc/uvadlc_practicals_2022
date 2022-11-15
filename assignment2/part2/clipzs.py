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

"""Zero-shot CLIP evaluation based on text-prompting."""
import os
import argparse
from pprint import pprint
import matplotlib.pyplot as plt

import torch
import numpy as np
from clip import clip

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from utils import AverageMeter, set_seed


DATASET = {"cifar10": CIFAR10, "cifar100": CIFAR100}


def parse_option():
    parser = argparse.ArgumentParser("Zero-shot CLIP")

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument(
        "--num_workers", type=int, default=16, help="num of workers to use"
    )

    # model
    parser.add_argument("--model", type=str, default="clip")
    parser.add_argument(
        "--arch", type=str, default="ViT-B/32", choices=["ViT-B/32", "ViT-B/16"]
    )

    # dataset
    parser.add_argument("--root", type=str, default="./data", help="dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="dataset",
        choices=["cifar10", "cifar100"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="dataset splits: (train/test)",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--test_noise",
        default=False,
        action="store_true",
        help="whether to add noise to the test images",
    )

    # input
    parser.add_argument(
        "--prompt_template", type=str, default="This is a photo of a {}"
    )
    parser.add_argument(
        "--class_names",
        nargs="+",
        type=str,
        default=None,
        help="(space separated) labels to use for the prompts; defaults to all classes in the dataset",
        # e.g. --class_names red blue green
    )

    # visualization
    parser.add_argument(
        "--visualize_predictions",
        default=False,
        action="store_true",
        help="whether to visualize the predictions of the first batch",
    )

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


class ZeroshotCLIP(nn.Module):
    """Module for zero-shot inference with CLIP."""

    def __init__(self, args, dataset, template):
        super(ZeroshotCLIP, self).__init__()

        self.device = args.device

        if args.class_names is None:
            classnames = dataset.classes
        else:
            classnames = args.class_names

        print(f"Using device: {args.device}")
        print(f"Loading CLIP (backbone: {args.arch})")
        clip_model = self.load_clip_to_cpu(args)
        clip_model.to(args.device)

        # if device is CPU, then make model as float() (This is a CLIP hack)
        if args.device == "cpu":
            clip_model = clip_model.float()

        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        print()
        print()
        print("List of prompts:")
        pprint(prompts)

        print("Precomputing text features")
        text_features = self.precompute_text_features(clip_model, prompts, args.device)

        self.class_names = classnames
        self.text_features = text_features
        self.clip_model = clip_model
        self.logit_scale = self.clip_model.logit_scale.exp().detach()

    def precompute_text_features(self, clip_model, prompts, device):
        """
        Precomputes text features for the given prompts.

        Args:
            clip_model (nn.Module): CLIP model
            prompts (list): list of prompts (strings)
            device (str): device to use for computation

        Returns:
            torch.Tensor: text features of shape (num_prompts, 512)

        Note: Do not forget to set torch.no_grad() while computing the text features.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Implement the precompute_text_features function.

        # Instructions:
        # - Given a list of prompts, compute the text features for each prompt.

        # Steps:
        # - Tokenize each text prompt using CLIP's tokenizer.
        # - Compute the text features (encodings) for each prompt.
        # - Normalize the text features.
        # - Return a tensor of shape (num_prompts, 512).

        # Hint:
        # - Read the CLIP API documentation for more details:
        #   https://github.com/openai/CLIP#api

        # remove this line once you implement the function
        raise NotImplementedError("Implement the precompute_text_features function.")

        #######################
        # END OF YOUR CODE    #
        #######################

    def model_inference(self, image):
        """
        Performs inference on a single image.

        Args:
            image (torch.Tensor): image tensor of shape (3, 224, 224)

        Returns:
            logits (torch.Tensor): logits of shape (num_classes,)
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Implement the model_inference function.

        # Instructions:
        # - Given an image, perform the forward pass of the CLIP model,
        #   i.e., compute the logits w.r.t. each of the prompts defined earlier.

        # Steps:
        # - Compute the image features (encodings) using the CLIP model.
        # - Normalize the image features.
        # - Compute similarity logits between the image features and the text features.
        #   You need to multiply the similarity logits with the logit scale (clip_model.logit_scale).
        # - Return logits of shape (num_classes,).

        # Hint:
        # - Read the CLIP API documentation for more details:
        #   https://github.com/openai/CLIP#api

        # remove this line once you implement the function
        raise NotImplementedError("Implement the model_inference function.")

        #######################
        # END OF YOUR CODE    #
        #######################

    def load_clip_to_cpu(self, args):
        """Loads CLIP model to CPU."""
        backbone_name = args.arch
        url = clip._MODELS[backbone_name]
        model_path = clip._download(url, args.root)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip.build_model(state_dict or model.state_dict())
        return model

    def num_params(self):
        """Prints number of parameters in the model."""
        print(
            "Model parameters:",
            f"{np.sum([int(np.prod(p.shape)) for p in self.clip_model.parameters()]):,}",
        )


def load_dataset(dataset, root, split, preprocess):
    """
    Loads a dataset object. Currently only supports CIFAR10 and CIFAR100.

    Args:
        dataset (str): name of the dataset
        root (str): path to the dataset
        split (str): train/test split
        preprocess (callable): preprocessing function for the dataset

    Returns:
        dataset (torch.utils.data.Dataset): dataset object
    """
    if dataset not in DATASET:
        raise ValueError(f"{dataset} is not supported. Choose among {DATASET.keys()}")
    is_train_split = split == "train"
    dataset = DATASET[dataset](
        root, transform=preprocess, download=True, train=is_train_split
    )
    return dataset


def visualize_predictions(images, logits, classnames, fig_file):
    """
    Visualizes the predictions of the model.

    Args:
        images (torch.Tensor): batch of images
        logits (torch.Tensor): logits of shape (batch_size, num_classes)
        classnames (list): list of classnames
        fig_file (str): path to save the figure
        topk (int): number of top predictions to visualize
    """
    # unnormalize images
    images = images.cpu()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    images = images * std[:, None, None] + mean[:, None, None]

    # get top predictions & apply softmax
    topk = min(5, logits.shape[1])
    probs, preds = logits.detach().topk(topk, dim=1)
    probs = probs.softmax(dim=1)

    # convert to numpy to use with plt
    preds = preds.cpu().numpy()
    probs = probs.cpu().numpy()

    # plot images
    h = int(np.floor(images.shape[0] / 2))
    plt.figure(figsize=(3.5 * h, 2 * h))
    for i, image in enumerate(images):
        # display image
        plt.subplot(h, 4, 2 * i + 1)
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.axis("off")

        # display topk predictions
        plt.subplot(h, 4, 2 * i + 2)
        k_range = np.arange(topk)
        plt.barh(k_range, probs[i])
        plt.yticks(
            k_range,
            [f"{classnames[p]} ({probs[i][k]:.2f})" for p, k in zip(preds[i], k_range)],
        )
        plt.xticks(np.arange(0, 1.1, 0.5))
        plt.gca().invert_yaxis()
        plt.grid(axis="x")

    plt.tight_layout()
    plt.savefig(fig_file)


def main():
    # Part 0.0: Read options from command line & fix seed
    args = parse_option()
    device = args.device
    set_seed(args.seed)

    # Part 0.1: set number of workers to max the number of CPU cores
    args.num_workers = min(args.num_workers, os.cpu_count())

    # Part 1. Load dataset and create dataloader
    _, preprocess = clip.load(args.arch)
    dataset = load_dataset(args.dataset, args.root, args.split, preprocess)

    loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Part 2. Initialize the inference class: ZeroshotCLIP
    print("Using prompt template:", args.prompt_template)
    clipzs = ZeroshotCLIP(args=args, dataset=dataset, template=args.prompt_template)

    # define the metric tracker for top1 accuracy
    top1 = AverageMeter("Acc@1", ":6.2f")

    # Part 3. (Optional) Visualize predictions
    if args.visualize_predictions:
        num_viz = 8
        idx = np.random.choice(len(dataset), num_viz, replace=False)
        images = [dataset[i][0] for i in idx]
        images = torch.stack(images).to(device)
        logits = clipzs.model_inference(images)

        c_names = ",".join(args.class_names) if args.class_names else "default"
        fig_file = f"{args.dataset}-{args.split}_{c_names}.png"
        visualize_predictions(images, logits, clipzs.class_names, fig_file)

    if args.class_names is not None:
        # No point in running evaluation if we don't use the dataset's class names
        return

    # Part 4. Inference loop
    print(f"Iterating over {args.split} set of {args.dataset}")

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Implement the inference loop

    # Steps:
    # - Iterate over the dataloader
    # - For each image in the batch, get the predicted class
    # - Update the accuracy meter

    # Hints:
    # - Before filling this part, you should first complete the ZeroShotCLIP class
    # - Updating the accuracy meter is as simple as calling top1.update(accuracy, batch_size)
    # - You can use the model_inference method of the ZeroshotCLIP class to get the logits

    # you can remove the following line once you have implemented the inference loop
    raise NotImplementedError("Implement the inference loop")

    #######################
    # END OF YOUR CODE    #
    #######################

    print(
        f"Zero-shot CLIP top-1 accuracy on {args.dataset}/{args.split}: {top1.val*100}"
    )


if __name__ == "__main__":
    main()
