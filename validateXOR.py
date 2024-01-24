"""
Author: Katelyn Van Dyke
Date: 23 Jan 2024
Course: Neural Networks
Professor: Dr. Derek T. Anderson
Description: 2d xor approximation with pytorch
"""

import torch
from XORModel import XORModel

# Initialize the model from file
sigmoid_activation = torch.nn.Sigmoid()
mse_loss = torch.nn.MSELoss()
model = XORModel(activation_fn=sigmoid_activation, loss_fn=mse_loss)
model.load_state_dict(torch.load('trained_model.pth'))

# Set the model to evaluation mode
model.eval()

# Validation tensors for xor
xor_0_0 = torch.tensor([[0, 0]], dtype=torch.float32)
xor_0_1 = torch.tensor([[0, 1]], dtype=torch.float32)
xor_1_0 = torch.tensor([[1, 0]], dtype=torch.float32)
xor_1_1 = torch.tensor([[1, 1]], dtype=torch.float32)

# Set validation threshold
threshold = 0.5

# Make raw predictions
with torch.no_grad():
    raw_predictions_0_0 = model(xor_0_0)
    raw_predictions_0_1 = model(xor_0_1)
    raw_predictions_1_0 = model(xor_1_0)
    raw_predictions_1_1 = model(xor_1_1)

# Convert predictions to binary output
binary_predictions_0_0 = (raw_predictions_0_0 > threshold).int().item()
binary_predictions_0_1 = (raw_predictions_0_1 > threshold).int().item()
binary_predictions_1_0 = (raw_predictions_1_0 > threshold).int().item()
binary_predictions_1_1 = (raw_predictions_1_1 > threshold).int().item()

# Print the results
print("Binary Predictions for XOR(0,0):", binary_predictions_0_0)
print("Binary Predictions for XOR(0,1):", binary_predictions_0_1)
print("Binary Predictions for XOR(1,0):", binary_predictions_1_0)
print("Binary Predictions for XOR(1,1):", binary_predictions_1_1)
