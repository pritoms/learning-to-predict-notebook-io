import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def train(input_tensor, target_tensor, model, optimizer, criterion):
    # Initialize hidden state
    hidden = model.initHidden()

    # Zero gradients
    optimizer.zero_grad()

    # Initialize loss
    loss = 0

    # Iterate through each input in input tensor
    for i in range(input_tensor.size(0)):
        # Get input and target
        input = input_tensor[i]
        target = target_tensor[i]

        # Forward pass
        output, hidden = model(input, hidden)

        # Calculate loss
        loss += criterion(output, target.unsqueeze(0))

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    return loss.item() / input_tensor.size(0)
