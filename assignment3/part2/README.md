# Assignment 3, Part 2: Adversarial Autoencoders

This folder contains the template code for implementing your own AAE model. This corresponds to Question 2.3 to 2.7 in the assignment. We will train the model on generating MNIST images. The code is structured in the following way:

* `mnist.py`: Contains a function for preparing the dataset and providing a data loader for training.
* `models.py`: Contains template classes for the Encoder, Decoder, Discriminator and the overall Adversarial Autoencoder.
* `train.py`: Contains the overall procedure of assignment, it parses terminal commands by user, then it sets the hyper-parameters, load dataset, initialize the model, set the optimizers and then
  it trains the adversarial auto-encoder and saves the network generations for each epoch.   
* `unittests.py`: Contains unittests for the Encoder, Decoder, Discriminator networks. It will hopefully help you debugging your code. Your final code should pass these unittests.
* `utils.py`: Contains logging utilities for Tensorboard.

A lot of code is already provided to you. Try to familiarize yourself with the code structure before starting your implementation. 
Your task is to fill in the missing code pieces (indicated by `NotImplementedError` or warnings printed). The main missing pieces are:

* In `models.py`, you need to implement the Encoder, Decoder, Discriminator and the overall Adversarial Autoencoder network. We suggested a base network architecture but feel free to experiment with different architectures. Also in Adversarial Autoencoder you have to design the required losses in the function `get_loss_discriminator` and `get_loss_autoencoder`.
* In `train.py`, you need to complete the `main` function, so that it defines the required optimizers. In function `train_aae` you should first calculate the discriminator and autoencoder losses.
  Then perform one optimization step for both autoencoder and discriminator parts. We have already provided additional logging utilities in the code to aid you for the questions, but feel free to replace or add your own logging codes.
  
Default hyper-parameters are provided in the `ArgumentParser` object of the respective training functions. Feel free to play around with those to familiarize yourself with the effect of different hyper-parameters. Nevertheless, your model should be able to generate decent images with the default hyper-parameters.
The training time with the default hyper-parameters and architecture is less than 30 minutes on a NVIDIA GTX1080Ti (GPU provided on Lisa).

