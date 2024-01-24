"""
Author: Katelyn Van Dyke
Date: 23 Jan 2024
Course: Neural Networks
Professor: Dr. Derek T. Anderson
Description: 2d xor approximation with pytorch
"""

import torch


class XORModel(torch.nn.Module):

    def __init__(self, activation_fn=None, loss_fn=None):
        super(XORModel, self).__init__()
        self.dense_layer = torch.nn.Linear(2, 2)
        self.output_layer = torch.nn.Linear(2, 1)
        # Set default activation function to Sigmoid if not provided
        self.activation_fn = activation_fn if activation_fn is not None else torch.nn.Sigmoid()
        # Set default loss function to Mean Squared Error if not provided
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()

    def forward(self, x):
        x = self.activation_fn(self.dense_layer(x))
        x = self.activation_fn(self.output_layer(x))
        return x
