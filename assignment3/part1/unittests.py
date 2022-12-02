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

import unittest
import numpy as np
import torch
import torch.nn as nn

from utils import sample_reparameterize, KLD, elbo_to_bpd
from cnn_encoder_decoder import CNNEncoder, CNNDecoder
import train_torch
import train_pl

"""
The following variables determine which training file to check.
- Set TEST_LIGHTNING to True if you are using train_pl.py
- Set TEST_TORCH to True if you are using train_torch.py
"""
TEST_LIGHTNING = False
TEST_TORCH = False

if not (TEST_LIGHTNING or TEST_TORCH):
    raise ValueError("Set either TEST_LIGHTNING or TEST_TORCH to True!")


class TestKLD(unittest.TestCase):

    @torch.no_grad()
    def test_normal_dist(self):
        mean = torch.zeros(1,1)
        log_std = torch.zeros(1,1)
        out = KLD(mean, log_std).numpy()
        self.assertTrue((out == 0).all(),
                         msg="The KLD for a normal distribution with mean 0 and std 1 must be 0, but is %s" % (str(out[0])))

    @torch.no_grad()
    def test_symmetry(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(5):
            mean = torch.randn(16,4)
            log_std = torch.randn(16,4)
            out1 = KLD(mean, log_std).numpy()
            out2 = KLD(-mean, log_std).numpy()
            self.assertTrue((out1 == out2).all(),
                            msg="The KLD must be symmetric for the mean values.\n"+\
                                "Positive mean:%s\nNegative mean:%s" % (str(out1), str(out2)))

    @torch.no_grad()
    def test_multivariate(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(5):
            mean = torch.randn(2,4)
            log_std = torch.randn(2,4)
            out = KLD(mean, log_std).numpy()
            out_manual = sum([KLD(mean[:,i:i+1], log_std[:,i:i+1]).numpy() for i in range(mean.shape[1])])
            self.assertTrue(len(out.shape) == 1,
                            msg="The KLD should be applied on a multi-variate Gaussian distribution.\n"+\
                                "Expected output shape: %s\nObtained output shape: %s" % (str(mean.shape[:-1]), str(out.shape)))
            self.assertTrue((out == out_manual).all(),
                            msg="The KLD should be applied on a multi-variate Gaussian distribution. You might be missing a sum."+\
                                "Expected output: %s\nObtained output: %s" % (str(out_manual), str(out)))

    @torch.no_grad()
    def test_mean_std(self):
        np.random.seed(42)
        torch.manual_seed(42)
        mean = torch.zeros(1,1) + 2
        log_std = torch.zeros(1,1)
        true_out = 2
        out = KLD(mean, log_std).squeeze().item()
        self.assertLess(abs(out - true_out), 1e-5,
                        msg="The terms of the mean and std in the KLD are not correct.\n" +
                            "Expected result: %f\nObtained result: %f" % (true_out, out))

    @torch.no_grad()
    def test_std(self):
        np.random.seed(42)
        torch.manual_seed(42)
        mean = torch.zeros(1,1)
        log_std = torch.zeros(1,1) + 1
        true_out = 2.194528
        out = KLD(mean, log_std).squeeze().item()
        self.assertLess(abs(out - true_out), 1e-5,
                        msg="The terms of the std in the KLD are not correct.\n" +
                            " Expected result: %f\nObtained result: %f" % (true_out, out))


class TestReparameterization(unittest.TestCase):

    def test_gradients(self):
        np.random.seed(42)
        torch.manual_seed(42)
        mean = torch.randn(16,4, requires_grad=True)
        log_std = torch.randn(16,4, requires_grad=True)
        out = sample_reparameterize(mean, log_std.exp())
        try:
            out.sum().backward()
        except RuntimeError:
            assert False, "The output tensor of reparameterization does not include the mean and std tensor in the computation graph."
        self.assertTrue(mean.grad is not None,
                         msg="Gradients of the mean tensor are None")
        self.assertTrue(log_std.grad is not None,
                         msg="Gradients of the standard deviation tensor are None")

    @torch.no_grad()
    def test_distribution(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(10):
            mean = torch.randn(1,)
            std = torch.randn(1,).exp()
            mean, std = mean.expand(20000,), std.expand(20000,)
            out = sample_reparameterize(mean, std)
            out_mean = out.mean()
            out_std = out.std()
            self.assertLess((out_mean - mean[0]).abs(), 1e-1,
                            msg="Sampled distribution does not match the mean.")
            self.assertLess((out_std - std[0]).abs(), 1e-1,
                            msg="Sampled distribution does not match the standard deviation.")


class TestBPD(unittest.TestCase):

    @torch.no_grad()
    def test_random_image_shape(self):
        np.random.seed(42)
        torch.manual_seed(42)
        for test_num in range(10):
            img = torch.zeros(8, *[np.random.randint(10,20) for _ in range(3)]) + 0.5
            elbo = -np.log(img).sum(dim=(1,2,3))
            bpd = elbo_to_bpd(elbo, img.shape)
            self.assertTrue(((bpd - 1.0).abs() < 1e-5).all(),
                             msg="The bits per dimension score for a random image has to be 1. Given scores: %s" % str(bpd))



class TestCNNEncoderDecoder(unittest.TestCase):

    @torch.no_grad()
    def test_encoder(self):
        np.random.seed(42)
        torch.manual_seed(42)
        skip_test = False
        try:
            enc = CNNEncoder()
        except NotImplementedError:
            skip_test = True

        if not skip_test:
            all_means, all_log_std = [], []
            for test_num in range(10):
                z_dim = np.random.randint(2,40)
                encoder = CNNEncoder(z_dim=z_dim)
                img = torch.randint(16, (32, 1, 28, 28))
                mean, log_std = encoder(img)
                self.assertTrue((mean.shape[0] == 32 and mean.shape[1] == z_dim),
                                 msg="The shape of the mean output should be batch_size x z_dim")
                self.assertTrue((log_std.shape[0] == 32 and log_std.shape[1] == z_dim),
                                 msg="The shape of the log_std output should be batch_size x z_dim")
                all_means.append(mean.reshape(-1))
                all_log_std.append(log_std.reshape(-1))
            means = torch.cat(all_means, dim=0)
            log_std = torch.cat(all_log_std, dim=0)
            self.assertTrue((means > 0).any() and (means < 0).any(), msg="Only positive or only negative means detected. Are you sure this is what you want?")
            self.assertTrue((log_std > 0).any() and (log_std < 0).any(), msg="Only positive or only negative log-stds detected. Are you sure this is what you want?")

    @torch.no_grad()
    def test_decoder(self):
        np.random.seed(42)
        torch.manual_seed(42)
        skip_test = False
        try:
            enc = CNNDecoder()
        except NotImplementedError:
            skip_test = True

        if not skip_test:
            z_dim = 20
            decoder  = CNNDecoder(z_dim=20)
            z = torch.randn(64, z_dim)
            imgs = decoder(z)
            self.assertTrue(len(imgs.shape) == 4 and all([imgs.shape[i] == o for i,o in enumerate([64,16,28,28])]),
                             msg="Output of the decoder should be an image with shape [B,C,H,W], but got: %s." % str(imgs.shape))
            self.assertTrue((imgs < 0).any(),
                             msg="The output of the decoder does not have any negative values. " + \
                                 "You might be applying a softmax on the decoder output. " + \
                                 "It is recommended to work on logits instead as this is numercially more stable. " + \
                                 "Ensure that you are using the correct loss accordingly.")


class TestVAE(unittest.TestCase):

    @torch.no_grad()
    def test_forward(self):
        np.random.seed(42)
        torch.manual_seed(42)
        if TEST_LIGHTNING:
            VAEClass = train_pl.VAE
        elif TEST_TORCH:
            VAEClass = train_torch.VAE
        else:
            print("TestVAE skipped as no train flag has been selected.")
            return

        vae = VAEClass(num_filters=32,
                       z_dim=20,
                       lr=1e-3)
        for p in vae.parameters():
            p.data.fill_(0.)

        rand_img = torch.randint(low=0, high=16, size=(4, 1, 28, 28)).long()
        L_rec, L_reg, bpd = vae(rand_img)

        # Testing shapes
        self.assertTrue(len(L_rec.squeeze().shape) == 0,
                        msg="The L_rec output must be a scalar, but has the shape %s." % str(bpd.shape))
        self.assertTrue(len(L_reg.squeeze().shape) == 0,
                        msg="The L_reg output must be a scalar, but has the shape %s." % str(bpd.shape))
        self.assertTrue(len(bpd.squeeze().shape) == 0,
                        msg="The BPD output must be a scalar, but has the shape %s." % str(bpd.shape))

        # Testing values
        true_L_rec = -np.log(0.0625)*784
        self.assertLessEqual(abs(L_rec.item() - true_L_rec), 1e-2,
                             msg="The L_rec output for zero-initialized networks must be %f, but is %f." % (true_L_rec, L_rec.item()))
        true_L_reg = 0.0
        self.assertLessEqual(abs(L_reg.item() - true_L_reg), 1e-5,
                             msg="The L_reg output for zero-initialized networks must be %f, but is %f." % (true_L_reg, L_reg.item()))
        true_bpd = 4.0
        self.assertLessEqual(abs(bpd.item() - true_bpd), 1e-5,
                             msg="The BPD output for zero-initialized networks must be %f, but is %f." % (true_bpd, bpd.item()))




if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKLD)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestReparameterization)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestBPD)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestCNNEncoderDecoder)
    unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestVAE)
    unittest.TextTestRunner(verbosity=2).run(suite)


