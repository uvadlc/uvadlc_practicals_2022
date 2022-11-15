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

"""Defines the trainer class for prompt-learning using CLIP."""
import os
from pprint import pprint
import torch
import torch.nn as nn
import numpy as np
import random
from clip import clip
from torch.cuda.amp import GradScaler
import time


from tqdm import tqdm
from vpt_model import CustomCLIP
from utils import cosine_lr, AverageMeter, ProgressMeter, accuracy, save_checkpoint, set_seed
from dataset import load_dataset, construct_dataloader


class Learner:
    """Trainer for prompt-learning using CLIP."""

    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.best_acc1 = 0

        # Load clip image transformation
        _, preprocess = clip.load(args.arch)

        self.train_dataset, self.val_dataset, self.test_dataset = load_dataset(
            args, preprocess
        )
        self.train_loader = construct_dataloader(args, self.train_dataset)
        self.val_loader = construct_dataloader(args, self.val_dataset)
        self.test_loader = construct_dataloader(args, self.test_dataset)

        PROMPT_TEMPLATE = args.text_prompt_template

        print("Building custom CLIP")
        self.vpt = CustomCLIP(args, self.test_dataset, template=PROMPT_TEMPLATE)

        # Optionally resume from a checkpoint
        if self.args.resume:
            self.resume_checkpoint()

        print("Turning off gradients in both the image and the text encoder")
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # TODO: Turn off gradients in both the image and the text encoder
        # Note: You need to keep the visual prompt's parameters trainable
        # Hint: Check for "prompt_learner" in the parameters' names

        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################

        # Double check
        enabled = set()
        for name, param in self.vpt.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated:")
        pprint(f"Parameters to be updated: {enabled}")

        # Print number of parameters
        num_params = sum(p.numel() for p in self.vpt.parameters() if p.requires_grad)
        print("Number of prompt parameters: ", num_params)

        self.vpt.to(self.device)

        # Define criterion and optimizer
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.vpt.parameters()),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()

        # Define scheduler
        total_steps = len(self.train_loader) * args.epochs
        self.scheduler = cosine_lr(
            self.optimizer, args.learning_rate, args.warmup, total_steps
        )
        self.reproduceability(args)

    def resume_checkpoint(self):
        """Resumes training from a checkpoint."""

        if os.path.isfile(self.args.resume):
            print("=> loading checkpoint '{}'".format(self.args.resume))
            if self.args.gpu is None:
                checkpoint = torch.load(self.args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(self.args.gpu)
                checkpoint = torch.load(self.args.resume, map_location=loc)
            self.args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if self.args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(self.args.gpu)
            self.vpt.prompt_learner.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(self.args.resume))

    def reproduceability(self, args):
        """Fixes the seed for reproducibility."""
        if args.seed is not None:
            set_seed(args.seed)

    def run(self):
        """Runs training for the specified number of epochs."""
        epochs_since_improvement = 0

        for epoch in range(self.args.epochs):

            # Train for one epoch
            self.train_one_epoch(epoch)

            # Evaluate on validation set
            acc1 = self.evaluate()

            # Remember best acc@1 and save checkpoint
            is_best = acc1 > self.best_acc1
            self.best_acc1 = max(acc1, self.best_acc1)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.vpt.prompt_learner.state_dict(),
                    "best_acc1": self.best_acc1,
                    "optimizer": self.optimizer.state_dict(),
                },
                self.args,
                is_best=is_best,
            )

            if is_best:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                print(f"There's no improvement for {epochs_since_improvement} epochs.")

                if epochs_since_improvement >= self.args.patience:
                    print("The training halted by early stopping criterion.")
                    break

    def train_one_epoch(self, epoch):
        """
        Updates (prompt) parameters for one epoch.

        Args:
            epoch (int): current epoch number

        Returns:
            tuple: (train_loss averaged across batch, train_acc across batch)
        """
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, top1],
            prefix="Epoch: [{}]".format(epoch),
        )

        # Switch to train mode
        self.vpt.train()

        num_batches_per_epoch = len(self.train_loader)

        end = time.time()
        for i, (images, target) in enumerate(tqdm(self.train_loader)):

            # Measure data loading time
            data_time.update(time.time() - end)

            # Adjust learning rate
            step = num_batches_per_epoch * epoch + i
            self.scheduler(step)

            #######################
            # PUT YOUR CODE HERE  #
            #######################

            # TODO: Implement the training step for a single batch

            # Steps ( your usual training loop :) ):
            # - Set the gradients to zero
            # - Move the images/targets to the device
            # - Perform a forward pass (using self.vpt)
            # - Compute the loss (using self.criterion)
            # - Perform a backward pass
            # - Update the parameters

            raise NotImplementedError
            #######################
            # END OF YOUR CODE    #
            #######################

            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            self.vpt.logit_scale.data = torch.clamp(
                self.vpt.logit_scale.data, 0, 4.6052
            )

            # Measure accuracy
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                progress.display(i)

            if i % self.args.save_freq == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.vpt.prompt_learner.state_dict(),
                        "best_acc1": self.best_acc1,
                        "optimizer": self.optimizer.state_dict(),
                    },
                    self.args,
                )

        return losses.avg, top1.avg

    def evaluate(self, split="valid"):
        """Evaluates the model on the given `split` set and returns average accuracy."""
        batch_time = AverageMeter("Time", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1_prompt = AverageMeter("Prompt Acc@1", ":6.2f")
        loader = self.val_loader if split == "valid" else self.test_loader
        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, top1_prompt],
            prefix="Validate: ",
        )

        # Switch to evaluation mode
        self.vpt.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(tqdm(loader)):

                #######################
                # PUT YOUR CODE HERE  #
                #######################

                # TODO: Implement the evaluation step for a single batch

                # Steps ( your usual evaluation loop :) ):
                # - Move the images/targets to the device
                # - Forward pass (using self.vpt)
                # - Compute the loss (using self.criterion)

                raise NotImplementedError
                #######################
                # END OF YOUR CODE    #
                #######################

                # Measure accuracy and record loss
                acc1 = accuracy(output, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_prompt.update(acc1[0].item(), images.size(0))

                # Measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    progress.display(i)

            print(
                " * Prompt Acc@1 {top1_prompt.avg:.3f}".format(top1_prompt=top1_prompt)
            )

        return top1_prompt.avg
