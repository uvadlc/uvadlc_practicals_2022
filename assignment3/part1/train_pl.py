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

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from mnist import mnist
from cnn_encoder_decoder import CNNEncoder, CNNDecoder
from utils import *


class VAE(pl.LightningModule):

    def __init__(self, num_filters, z_dim, lr):
        """
        PyTorch Lightning module that summarizes all components to train a VAE.
        Inputs:
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
            lr - Learning rate to use for the optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder = CNNEncoder(z_dim=z_dim, num_filters=num_filters)
        self.decoder = CNNDecoder(z_dim=z_dim, num_filters=num_filters)

    def forward(self, imgs):
        """
        The forward function calculates the VAE-loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W].
                   The input images are converted to 4-bit, i.e. integers between 0 and 15.
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """

        # Hints:
        # - Implement the empty functions in utils.py before continuing
        # - The forward run consists of encoding the images, sampling in
        #   latent space, and decoding.
        # - By default, torch.nn.functional.cross_entropy takes the mean accross
        #   all axes. Do not forget to change the 'reduction' parameter to
        #   make it consistent with the loss definition of the assignment.

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        L_rec = None
        L_reg = None
        bpd = None
        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################
        return L_rec, L_reg, bpd

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled, 4-bit images. Shape: [B,C,H,W]
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        x_samples = None
        raise NotImplementedError
        #######################
        # END OF YOUR CODE    #
        #######################
        return x_samples

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("train_reconstruction_loss", L_rec, on_step=False, on_epoch=True)
        self.log("train_regularization_loss", L_reg, on_step=False, on_epoch=True)
        self.log("train_ELBO", L_rec + L_reg, on_step=False, on_epoch=True)
        self.log("train_bpd", bpd, on_step=False, on_epoch=True)

        return bpd

    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("val_reconstruction_loss", L_rec)
        self.log("val_regularization_loss", L_reg)
        self.log("val_ELBO", L_rec + L_reg)
        self.log("val_bpd", bpd)

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, bpd = self.forward(batch[0])
        self.log("test_bpd", bpd)


class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=64, every_n_epochs=5, save_to_disk=False):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated sample images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """
        samples = pl_module.sample(self.batch_size)
        samples = samples.float() / 15  # Converting 4-bit images to values between 0 and 1
        grid = make_grid(samples, nrow=8, normalize=True, value_range=(0, 1), pad_value=0.5)
        grid = grid.detach().cpu()
        trainer.logger.experiment.add_image("Samples", grid, global_step=epoch)
        if self.save_to_disk:
            save_image(grid,
                        os.path.join(trainer.logger.log_dir, f"epoch_{epoch}_samples.png"))


def train_vae(args):
    """
    Function for training and testing a VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """

    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = mnist(batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   root=args.data_dir)

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback = GenerateCallback(save_to_disk=True)
    save_callback = ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd")
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         callbacks=[save_callback, gen_callback],
                         enable_progress_bar=args.progress_bar)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    if not args.progress_bar:
        print("[INFO] The progress bar has been suppressed. For updates on the training " + \
              f"progress, check the TensorBoard file at {trainer.logger.log_dir}. If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible
    model = VAE(num_filters=args.num_filters,
                z_dim=args.z_dim,
                lr=args.lr)

    # Training
    gen_callback.sample_and_save(trainer, model, epoch=0)  # Initial sample
    trainer.fit(model, train_loader, val_loader)

    # Testing
    model = VAE.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=True)

    # Manifold generation
    if args.z_dim == 2:
        img_grid = visualize_manifold(model.decoder)
        save_image(img_grid,
                   os.path.join(trainer.logger.log_dir, 'vae_manifold.png'),
                   normalize=False)

    return test_result


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--z_dim', default=20, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--num_filters', default=32, type=int,
                        help='Number of channels/filters to use in the CNN encoder/decoder.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory where to look for the data. For jobs on Lisa, this should be $TMPDIR.')
    parser.add_argument('--epochs', default=80, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='VAE_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    train_vae(args)

