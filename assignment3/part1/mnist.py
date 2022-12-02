################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torchvision
from torchvision import transforms
import torch
import torch.utils.data as data
from torch.utils.data import random_split
import numpy as np

def discretize(x, num_values):
    return (x * num_values).long().clamp_(max=num_values-1)

def mnist(root='../data/', batch_size=128, num_workers=4, download=True):
    """
    Returns data loaders for 4-bit MNIST dataset, i.e. values between 0 and 15.

    Inputs:
        root - Directory in which the MNIST dataset should be downloaded. It is better to
               use the same directory as the part2 of the assignment to prevent duplicate
               downloads.
        batch_size - Batch size to use for the data loaders
        num_workers - Number of workers to use in the data loaders.
        download - If True, MNIST is downloaded if it cannot be found in the specified
                   root directory.
    """
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          transforms.Lambda(lambda x: discretize(x, num_values=16))
                                        ])

    dataset = torchvision.datasets.MNIST(
        root, train=True, transform=data_transforms, download=download)
    test_set = torchvision.datasets.MNIST(
        root, train=False, transform=data_transforms, download=download)

    train_dataset, val_dataset = random_split(dataset,
                                              lengths=[54000, 6000],
                                              generator=torch.Generator().manual_seed(42))

    # Each data loader returns tuples of (img, label)
    # For the generative models we don't need the labels, which we need to take into account
    # when writing the train code.
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True)
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)

    return train_loader, val_loader, test_loader

