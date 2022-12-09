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
# Author: Deep Learning Course | Fall 2022
# Date Created: 2022-11-25
################################################################################


import unittest
import numpy as np
import torch
import warnings
from models import AdversarialAE

warnings.filterwarnings("ignore")


class TestEncoder(unittest.TestCase):

    @torch.no_grad()
    def test_shape(self):
        np.random.seed(42)
        torch.manual_seed(42)
        z_dim = 10
        ae = AdversarialAE(z_dim=z_dim)
        x = torch.randn([4, 1, 28, 28])
        latent = ae.encoder(x)
        self.assertTrue(len(latent.shape) == 2 and
                        all([latent.shape[i] == o for i, o in enumerate([4, z_dim])]),
                        msg="The output of the encoder should be an image with shape [B, z_dim].")


class TestDecoder(unittest.TestCase):

    @torch.no_grad()
    def test_shape(self):
        np.random.seed(42)
        torch.manual_seed(42)
        z_dim = 10
        ae = AdversarialAE(z_dim=z_dim)
        z = torch.randn([4, z_dim])
        latent = ae.decoder(z)
        self.assertTrue(len(latent.shape) == 4 and
                        all([latent.shape[i] == o for i, o in enumerate([4, 1, 28, 28])]),
                        msg="The output of the decoder should be an image with shape [B,C,H,W].")

    @torch.no_grad()
    def test_output_values(self):
        np.random.seed(42)
        torch.manual_seed(42)
        z_dim = 10
        ae = AdversarialAE(z_dim=z_dim)
        z = torch.randn(128, z_dim) * 50
        imgs = ae.decoder(z)
        self.assertTrue((imgs >= -1).all() and (imgs <= 1).all(),
                        msg="The output of the autoencoder should have values between -1 and 1. " 
                            "A tanh as output activation function might be missing.")


class TestDiscriminator(unittest.TestCase):

    @torch.no_grad()
    def test_shape(self):
        np.random.seed(42)
        torch.manual_seed(42)
        z_dim = 10
        ae = AdversarialAE(z_dim=z_dim)
        z = torch.randn(128, z_dim)
        preds = ae.discriminator(z)
        self.assertTrue(len(preds.shape) == 2 and
                        all([preds.shape[i] == o for i, o in enumerate([128, 1])]),
                        msg="The output of the discriminator should be a prediction with shape [B, 1].")


class TestAAE(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        z_dim = 10
        AAE = AdversarialAE(z_dim=z_dim)
        for p in AAE.parameters():
            p.data.fill_(0.)
        self.aae = AAE

    def test_autoencoder_step(self):
        x_real = torch.zeros(4, 1, 28, 28, dtype=torch.float32)
        recon_batch, z_fake = self.aae(x_real)

        ae_loss, ae_dict = self.aae.get_loss_autoencoder(x_real, recon_batch, z_fake, lambda_=0)
        self.assertTrue(len(ae_loss.squeeze().shape) == 0,
                        msg="The loss must be a scalar, but has the shape %s." % str(ae_loss.shape))
        true_loss = -np.log(0.5)
        self.assertLessEqual(abs(ae_loss.item() - true_loss), 1e-4,
                             msg="The loss for zero-initialized networks must be %f, but is %f." % (
                             true_loss,
                             ae_loss.item()))
        ae_loss.backward()
        self.assertTrue(next(iter(self.aae.encoder.parameters())).grad is not None,
                        msg="No gradients detected for the generator in the generator_step.")
        self.aae.zero_grad()

    def test_reconstruction_step(self):
        x_real = torch.zeros(4, 1, 28, 28, dtype=torch.float32)
        recon_batch, z_fake = self.aae(x_real)

        ae_loss, ae_dict = self.aae.get_loss_autoencoder(x_real, recon_batch, z_fake, lambda_=1)

        self.assertTrue(len(ae_loss.squeeze().shape) == 0,
                        msg="The reconstruction loss must be a scalar, but has the shape %s."
                            "Something is wrong with either encoder or decoder." % str(ae_loss.shape))
        true_loss = 0
        self.assertTrue((abs(ae_loss.item() - true_loss) < 1e-4) or (abs(ae_loss.item() - 2 * true_loss) < 1e-4),
                        msg="The reconstruction loss for zero-initialized networks must be %f, but is %f."
                            "Something is wrong with either encoder or decoder." % (true_loss, ae_loss.item()))

        ae_loss.backward()
        self.assertTrue(next(iter(self.aae.encoder.parameters())).grad is not None,
                        msg="No gradients detected for the reconstruction in the reconstruction_step. "
                            "Something is wrong with either encoder or decoder.")
        self.aae.zero_grad()

    def test_discriminator_step(self):
        x_real = torch.zeros(4, 1, 28, 28, dtype=torch.float32)
        _, z_fake = self.aae(x_real)

        disc_loss, disc_dict = self.aae.get_loss_discriminator(z_fake)

        self.assertTrue(len(disc_loss.squeeze().shape) == 0,
                        msg="The discriminator loss must be a scalar, but has the shape %s." % str(disc_loss.shape))
        true_loss = -np.log(0.5)
        self.assertTrue((abs(disc_loss.item() - true_loss) < 1e-4) or (abs(disc_loss.item() - 2 * true_loss) < 1e-4),
                        msg="The discriminator loss for zero-initialized networks must be %f, but is %f." % (
                        true_loss, disc_loss.item()))

        disc_loss.backward()
        self.assertTrue(next(iter(self.aae.discriminator.parameters())).grad is not None,
                        msg="No gradients detected for the discriminator in the discriminator_step.")
        self.aae.zero_grad()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEncoder)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDecoder)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDiscriminator)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestAAE)
    unittest.TextTestRunner(verbosity=2).run(suite)
