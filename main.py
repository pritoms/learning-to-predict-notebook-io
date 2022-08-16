import sentencepiece as spm
import numpy as np
import json
import os
import nbformat as nbf
import torch as th
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from keras.preprocessing.sequence import pad_sequences
from model import RNN
from train import train
from utils import generate

# Build dataset
def build_dataset(directory):
    # Get all files in directory with .ipynb extension type 
    files = [f for f in os.listdir(directory) if f[-6:] == '.ipynb']

    # Initialize dataset
    dataset = []

    # Iterate through each file in directory
    for file in files:
        # Read notebook content
        with open(directory + '/' + file) as f:
            nb = nbf.read(f, as_version=4)

        # Iterate through each cell in notebook
        for cell in nb['cells']:
            # If cell is code type, add to dataset
            if cell['cell_type'] == 'code':
                dataset.append(cell['source'])

    # Shuffle dataset
    random.shuffle(dataset)

    # Split dataset into training and testing datasets
    split = int(len(dataset) * 0.8)

    return dataset[:split], dataset[split:]

# Tokenize dataset
def tokenize(dataset):
    # Initialize tokenized dataset
    tokenized_dataset = []

    # Iterate through each data in dataset
    for data in dataset:
        # Tokenize data and add to tokenized dataset
        tokenized_dataset.append(sp.EncodeAsIds(' '.join(data)))

    return np.array(tokenized_dataset)

# Pad dataset
def pad(dataset):
    return pad_sequences(dataset, maxlen=100, padding='post')

# Convert dataset to tensors
def to_tensors(dataset):
    dataset = th.from_numpy(dataset).long()
    return dataset, th.LongTensor([len(seq) for seq in dataset])

# Generate new notebook
def generate_new_notebook(model, spm):
    return spm.DecodeIds(generate(model, spm.PieceToId('<s>'), spm.PieceToId('</s>'), 100))

# Training dataset
train_data, test_data = build_dataset('notebook_training')

# Training sentencepiece model
spm.SentencePieceTrainer.Train('--input=train_data.txt --model_prefix=spm --vocab_size=116')

# Load sentencepiece model
sp = spm.SentencePieceProcessor()
sp.Load('spm.model')

# Tokenizing training dataset
train_data = tokenize(train_data)
test_data = tokenize(test_data)

# Padding training dataset
train_data = pad(train_data)
test_data = pad(test_data)

# Converting training dataset to tensors
train_data, train_lengths = to_tensors(train_data)
test_data, test_lengths = to_tensors(test_data)

# Initialize model
model = RNN(116, 256, 116)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize criterion
criterion = nn.CrossEntropyLoss()

# Initialize number of epochs
n_epochs = 100

# Iterate through each epoch
for epoch in range(1, n_epochs + 1):
    # Initialize loss
    loss = 0

    # Iterate through each data in train_data
    for i in range(train_data.size(0)):
        # Get input and target tensors
        input_tensor = train_data[i]
        target_tensor = th.cat((train_data[i][1:], train_data[i][0].unsqueeze(0)))

        # Train model
        loss += train(input_tensor, target_tensor, model, optimizer, criterion)

    # Print loss
    print('Epoch: %d Loss: %.4f' % (epoch, loss / train_data.size(0)))

# Generate new notebook
new_notebook = generate(model, sp.PieceToId('<s>'), sp.PieceToId('</s>'), 100)

# Decode new notebook
new_notebook = sp.DecodeIds(new_notebook)
