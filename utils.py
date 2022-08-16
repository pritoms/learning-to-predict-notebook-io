import sentencepiece as spm
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

def generate(model, start_token, end_token, max_length):
    # Initialize hidden state
    hidden = model.initHidden()

    # Initialize input tensor
    input = th.LongTensor([start_token])

    # Initialize output tensor
    output = []

    # Iterate through each token in output tensor
    for i in range(max_length):
        # Forward pass
        output_tensor, hidden = model(input, hidden)

        # Get top token from output tensor
        top_token = output_tensor.argmax(1)[0].item()

        # If top token is end token, break
        if top_token == end_token:
            break

        # Append top token to output tensor
        output.append(top_token)

        # Set input to top token
        input = th.LongTensor([top_token])

    return output
