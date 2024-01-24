"""
Author: Katelyn Van Dyke
Date: 23 Jan 2024
Course: Neural Networks
Professor: Dr. Derek T. Anderson
Description: 2d xor approximation with pytorch
"""

import torch


class XORModel(torch.nn.Module):

    def __init__(self, activation_fn, loss_fn):
        super(XORModel, self).__init__()
        self.dense_layer = torch.nn.Linear(2, 2)
        self.output_layer = torch.nn.Linear(2, 1)
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn

    def forward(self, x):
        x = self.activation_fn(self.dense_layer(x))
        x = self.activation_fn(self.output_layer(x))
        return x
