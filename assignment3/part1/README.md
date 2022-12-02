# Assignment 3, Part 1: Variational Autoencoders

This folder contains the template code for implementing your own VAE model. This corresponds to Questions 1.7 to 1.9 in the assignment.
We will train the model on generating 4-bit MNIST images. The original MNIST dataset contains images with pixel values between 0 and 1. To discretize those, we multiply pixel values with 16 and map the result to the closest integer value (rounding down 16 to 15). This is a 4-bit representation of the original image. Standard RGB images are usually using 8-bit encodings (i.e. values between 0 and 255), but to simplify the task, we only use 4 bits here.

The code is structured in the following way:
* `fmnist.py`: Contains a function for preparing the discretized dataset and providing a data loader for training, validation and testing.
* `cnn_encoder_decoder.py`: Contains template classes for the Encoder and Decoder based on an CNN.
* `train_pl.py`: Contains training functionalities such as the training loop, logging, saving, etc. We have provided you with logging utilities and general code structure so that you can focus on the important parts of the VAE model.
* `utils.py`: Contains functionalities that are required for training, such as the reparameterization trick, the KL divergence, bpd calculation and manifold generation.
* `unittests.py`: Contains unittests for the Encoder and Decoder networks, as well as functions of `utils.py`. It will hopefully help you debugging your code. Your final code should pass these unittests.

A lot of code is already provided to you. Try to familiarize yourself with the code structure before starting your implementation.
Your task is to fill in the missing code pieces (indicated by `NotImplementedError` or warnings printed). The main missing pieces are:
* In `utils.py`, you need to implement couple of smaller functions that are used in the training file:
  * [Q1.7] A `sample_reparameterize` function that should implement the reparameterization trick, i.e. sampling from a distribution with means and standard deviation that require gradients.
  * [Q1.7] A `KLD` function that should implement the KL divergence of the unit Gaussian prior and the predicted distribution.
  * [Q1.7] A `elbo_to_bpd` function that should implement the bits-per-dimension metric. The ELBO represents the negative log likelihood given by the ELBO objective of the VAE (reconstruction loss plus regularization loss).
  * [Q1.9] A `visualize_manifold` function that should implement the visualization of the manifold in latent space. See question 1.9 for details.
* In `train_pl.py`, you need to implement:
  * [Q1.7] A `forward` function which returns the reconstruction and regularization loss, as well as the bits per dimension metric for a single batch.
  * [Q1.8] A `sample` function that creates new images with the Decoder. We have already taken care of saving/logging those in the function `sample_and_save`.

Default hyperparameters are provided in the `ArgumentParser` object of the respective training functions. Feel free to play around with those to familiarize yourself with the effect of different hyperparameters. Nevertheless, your model should be able to generate decent images with the default hyperparameters.
  If you test the code on your local machine, you can use the argument `--progress_bar` to show a training progressbar. Remember to not use this on Lisa as it otherwise fills up your SLURM output file very quickly. It is recommended to look at the TensorBoard there instead.
  The training time with the default hyperparameters is less than 15 minutes on a NVIDIA GTX1080Ti (GPU provided on Lisa).

