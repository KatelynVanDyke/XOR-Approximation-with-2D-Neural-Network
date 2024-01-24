"""
Author: Katelyn Van Dyke
Date: 23 Jan 2024
Course: Neural Networks
Professor: Dr. Derek T. Anderson
Description: 2d xor approximation with pytorch
"""

import torch
from XORModel import XORModel


# XOR input and output
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Initialize model with activation and loss functions
sigmoid_activation = torch.nn.Sigmoid()
mse_loss = torch.nn.MSELoss()
model = XORModel(activation_fn=sigmoid_activation, loss_fn=mse_loss)

# Define learning rate, batch size, epochs, and optimizer for training
learning_rate = 0.1
batch_size = 4
epochs = 20000
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train model
for epoch in range(epochs):

    # Forward pass
    outputs = model(X)

    # Compute loss
    loss = model.loss_fn(outputs, y)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Perform accuracy check in 10% intervals
    if epoch % int(epochs * 0.1) == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
