# Learning to Predict Notebook IO

We will create a causal generative model that will learn to predict the cells of a given jupyter notebook (.ipynb extension types), as well as the interpreters response. The notebook will be divided into several sections from building dataset to generating new notebooks.

## Building the Dataset

We have a collection of `ipynb` files in the `notebook_training` directory. In this section, our objective is to read the contents of the notebook and build the training datasets.

### Reading Notebook Content

**Required Modules**

- `json`
- `os`
- `nbformat`

> **Note: In order to install `nbformat`, run `pip install nbformat`**

The `nbformat` module provides a convenient way to read the contents of a notebook.

**Reading Notebook Content**

```python
import nbformat as nbf

with open('notebook_training/notebook_1.ipynb') as f:
    nb = nbf.read(f, as_version=4)
```

The `nb` variable now contains the contents of the notebook. The `nb` variable is a dictionary with the following keys:
- `cells`: A list of cells in the notebook. Each cell is a dictionary with the following keys:
    - `cell_type`: The type of cell (e.g. code, markdown, etc.)
    - `source`: The source code for the cell. This is a list of strings, where each string is a line in the cell.
    - `metadata`: A dictionary containing metadata for the cell. This is an empty dictionary for most cells.
- `metadata`: A dictionary containing metadata for the notebook. This is an empty dictionary for most notebooks.
- `nbformat`: The version of the notebook format that was used to write this notebook file. This should be 4 for all notebooks written by Jupyter Notebook 4.0 and above.
- `nbformat_minor`: The minor version of the notebook format that was used to write this notebook file. This should be 0 for all notebooks written by Jupyter Notebook 4.0 and above.


### Building Dataset from Notebook Content

In this section, we will build our dataset from the contents of the notebook files in the `notebook_training` directory. We will use two datasets: one for training and one for testing our model. We will use 80% of our data for training and 20% for testing our model. 


**Required Modules**
- `json`
- `os`
- `nbformat`


**Building Dataset**


```python
import json, os, nbformat as nbf, random

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
```

**Building Datasets**

```python
train_data, test_data = build_dataset('notebook_training')
```


## Processing Datasets

Now that we have the training and testing data ready, we can proceed to the processing step, where we would turn our text data into suitable representations for our model.

### Tokenizing Datasets

In this section, we will tokenize our datasets. We will use the `sentencepiece` module to tokenize our datasets.

**Required Modules**
- `sentencepiece`
- `numpy`
- `torch`

> **Note: In order to install `sentencepiece`, run `pip install sentencepiece`**

**Training SentencePiece Model**

```python
import sentencepiece as spm

# Creating training data from our datasets
with open('train_data.txt', 'w') as f:
    for data in train_data:
        f.write(' '.join(data) + '\n')

# Training SentencePiece Model
spm.SentencePieceTrainer.Train('--input=train_data.txt --model_prefix=spm --vocab_size=116')
```

**Loading SentencePiece Model**

```python
sp = spm.SentencePieceProcessor()
sp.Load('spm.model')
```


**Tokenizing Datasets**

```python
import numpy as np, torch as th

def tokenize(dataset):
    # Initialize tokenized dataset
    tokenized_dataset = []

    # Iterate through each data in dataset
    for data in dataset:
        # Tokenize data and add to tokenized dataset
        tokenized_dataset.append(sp.EncodeAsIds(' '.join(data)))

    return np.array(tokenized_dataset)


train_data = tokenize(train_data)
test_data = tokenize(test_data)
```


### Padding Datasets

 In this section, we will pad our datasets to a fixed length of `100`. We will use the `pad_sequences` function from the `keras` module to pad our datasets. 

 **Required Modules**

 - `keras`

 > **Note: In order to install `keras`, run `pip install keras`**

 **Padding Datasets**

 ```python 
 from keras.preprocessing.sequence import pad_sequences

 train_data = pad_sequences(train_data, maxlen=100, padding='post')
 test_data = pad_sequences(test_data, maxlen=100, padding='post')

 ```

 ### Converting Datasets to Tensors

 In this section, we will convert our datasets to tensors using the `torch` module. We will also create a tensor for the lengths of each sequence in our datasets. This is required for the `pack_padded_sequence` function in the `torch.nn.utils.rnn` module, which we will use later on when training our model. 

 **Required Modules**

 - `torch`

 **Converting Datasets to Tensors**

 ```python
 train_data = th.from_numpy(train_data).long()
 test_data = th.from_numpy(test_data).long()

 train_lengths = th.LongTensor([len(seq) for seq in train_data])
 test_lengths = th.LongTensor([len(seq) for seq in test_data])
 ```

 ## Building the Model

In this section, we will build our model. We will use a recurrent neural network (RNN) as our model. We will use the `torch.nn` module to build our model. 

**Required Modules**
- `torch`
- `torch.nn`
- `torch.nn.utils.rnn`

**Building the Model**

```python
import torch.nn as nn, torch.nn.utils.rnn as rnn_utils

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden

    def initHidden(self):
        return th.zeros(1, 1, self.hidden_size)
```


## Training the Model

In this section, we will train our model using the `train_data` and `train_lengths` tensors that we created earlier on. We will use the `torch.optim` module to train our model. 

**Required Modules**
- `torch`
- `torch.nn`
- `torch.nn.utils.rnn`
- `torch.optim`


**Training Helper Methods**

```python
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
```

**Training the Model**

```python
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
```


## Generating New Notebooks

In this section, we will generate new notebooks using our trained model. We will use the `torch.nn.utils.rnn` module to generate new notebooks. 

**Required Modules**
- `torch`
- `torch.nn`
- `torch.nn.utils.rnn`

**Generating New Notebooks**

```python
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
```

**Generating New Notebooks**

```python
# Generate new notebook
new_notebook = generate(model, sp.PieceToId('<s>'), sp.PieceToId('</s>'), 100)

# Decode new notebook
new_notebook = sp.DecodeIds(new_notebook)
```
