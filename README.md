# XOR-Approximation-with-2D-Neural-Network

## Overview

This project implements a simple XOR model using PyTorch.

## Requirements

- Python 3.x
- PyTorch

## Model Architecture

The model architecture consists of a fully connected layer with 2 neurons followed by a sigmoid activation, and an output layer with a sigmoid activation. The architecture is defined in the `XORModel` class.
* 2 feature input as pytorch tensor
* 1 dense (fully connected) layer with 2 neurons
* Sigmoid activation
* 1 feature output as pytorch tensor
* Mean Squared Error loss evaluation
* Binary prediction using 0.5 decision threshold

## Training

The model is trained using Pytorch tensors representing the XOR logical operation. The training loop and hyperparameters are provided in trainXOR.py. The trained model parameters are saved to `trained_model.pth`.

## Validation

The model can be validated with the same XOR tensors used in training. The validateXOR.py script loads the trained model and prints binary predictions for each input.

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/KatelynVanDyke/XOR-Approximation-with-2D-Neural-Network.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Train the model:

    ```bash
    python trainXOR.py
    ```

4. Evaluate the model using validation script:

    ```bash
    python validateXOR.py
    ```

5. (Optional) Experiment by modifying the activation function, loss function, learning rate, batch size, epochs (number of training iterations), and decision threshold. If not specified, the model will use Sigmoid activation and Mean Squared Error loss.

## License

This project is licensed under the [MIT License](LICENSE).
