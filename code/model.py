import math
import numpy as np
import os
import torch

class generator():
    def __init__(self, input_size):
        super(generator, self).__init__()

        pass

    def call(self, inputs):
        """
        Performs a forward pass for the GAN
        """

        pass

    def loss_function(x_hat, x, mu, logvar):
        """
        Computes the average loss of the GAN in the current generated example.
        """

        pass



class discriminator():
    def __init__(self, input_size):
        super(discriminator, self).__init__()

        pass

    def call(self, inputs):
        """
        Performs a forward pass for the GAN
        """

        pass

    def loss_function(x_hat, x, mu, logvar):
        """
        Computes the average loss of the GAN in the current generated example.
        """

        pass
