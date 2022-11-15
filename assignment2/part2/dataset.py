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

"""Defines helper functions for loading data and constructing dataloaders."""
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, random_split
import torch

DATASET = {"cifar10": CIFAR10, "cifar100": CIFAR100}


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # TODO: Given a batch of images, add Gaussian noise to each image.

        # Hints:
        # - You can use torch.randn() to sample z ~ N(0, 1).
        # - Then, you can transform z s.t. it is sampled from N(self.mean, self.std)
        # - Finally, you can add the noise to the image.

        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_dataset(args, preprocess):
    train_transform = preprocess
    if args.test_noise:
        test_transform = Compose(preprocess.transforms + [AddGaussianNoise()])
    else:
        test_transform = preprocess

    train_dataset = DATASET[args.dataset](
        args.root, transform=train_transform, download=True, train=True
    )

    ratio = 0.2
    valid_size = int(len(train_dataset) * ratio)
    train_size = int(len(train_dataset) - valid_size)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, valid_size])

    test_dataset = DATASET[args.dataset](
        args.root, transform=test_transform, download=True, train=False
    )

    return train_dataset, val_dataset, test_dataset


def construct_dataloader(args, dataset):
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
