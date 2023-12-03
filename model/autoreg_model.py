import torch
import torch.nn as nn

import numpy as np

from loader.loader import CodonDataset


class AutoregressiveRNNModel(nn.Module):
    def __init__(self, layer_type, input_size, output_size, hidden_dim, n_layers):
        super(AutoregressiveRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.layer_type = layer_type

        # defining layers
        if layer_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        if layer_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        else:
            # default vanilla rnn
            self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # fully connected layer -> connect network to output labels
        self.activation = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        batch_size = x.size(0)

        # Initializing hidden state for first input
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        # Special case for LSTM
        if self.layer_type == "LSTM":
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
            hidden = (hidden, c0)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden_n = self.rnn(x, hidden)

        out = self.activation(out)
        out = self.fc(out)
        out = self.softmax(out)

        # return output layer and hidden state (for training and RNN optimization)
        return out, hidden_n

    def train(self, optimizer, criterion, train_loader):
        epoch_loss = 0
        num_batches = 0
        for x, y in train_loader:
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            output, hidden = self(x)
            output = output.permute(0,2,1)
            y = y.permute(0,2,1)
            loss = criterion(output, y)
            loss.backward()  # Does backpropagation and calculates gradients
            epoch_loss += loss.item()
            optimizer.step()  # Updates the weights accordingly
            # Calculate accuracy
            predicted_labels = torch.argmax(output, dim=0)
            correct_predictions = (
                (predicted_labels == y).sum().item()
            )  
            total_samples = np.prod(np.array(y.shape))
            accuracy = correct_predictions / total_samples
            # Calculate num_batches
            num_batches += 1
        epoch_loss /= num_batches
        return epoch_loss, accuracy, total_samples

    def eval(self, criterion, data_loader, predict=False):
        # accuracy and loss on test set
        test_loss = 0
        correct = 0
        total = 0
        test_accuracy = 0
        test_loss = 0
        with torch.no_grad():
            for x, y in data_loader:
                outputs, hidden= self(x)
                outputs = outputs.permute(0,2,1)
                y = y.permute(0,2,1)
                loss = criterion(outputs, y)
                test_loss += loss.item()
                #TODO: look into predict method
                #predicted = self.predict(x)
                predicted = torch.argmax(outputs, dim=0)
                total += np.prod(np.array(y.shape))
                correct += (predicted == y ).sum().item()
                test_accuracy = correct / total
            test_loss /= len(data_loader)

        return test_loss, test_accuracy, total
