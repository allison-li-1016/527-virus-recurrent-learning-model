import torch
import torch.nn as nn

import numpy as np


class RNNModel(nn.Module):
    def __init__(self, layer_type, input_size, output_size, hidden_dim, n_layers):
        super(RNNModel, self).__init__()
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
        self.fc = nn.Linear(hidden_dim, output_size)

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

        out = self.fc(out)

        # return output layer and hidden state (for training and RNN optimization)
        return out, hidden_n

    def train(self, optimizer, criterion, epochs, train_loader, verbose=True):
        epoch_losses = []
        epoch_resolution = max(1, epochs // 10)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            num_batches = 0
            for x, y in train_loader:
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                output, hidden = self(x)
                loss = criterion(output, y)
                loss.backward()  # Does backpropagation and calculates gradients
                epoch_loss += loss.item()
                optimizer.step()  # Updates the weights accordingly

                # Calculate accuracy
                predicted_labels = torch.argmax(output, dim=0)
                correct_predictions = (
                    (predicted_labels == y).sum().item()
                )  # Add in check for codon = NNN
                total_samples = np.prod(np.array(y.shape))
                accuracy = correct_predictions / total_samples
                # Calculate num_batches
                num_batches += 1
            epoch_losses.append(epoch_loss / num_batches)

            if not verbose:
                return epoch_losses

            if epoch % epoch_resolution == 0:
                print("Epoch: {}/{}.............".format(epoch, epochs), end=" ")
                print(
                    "Loss: {:.4f}, Accuracy: {:.2f}%".format(
                        loss.item(), accuracy * 100
                    )
                )
                print("correct labels: " + str(correct_predictions))
                print("total samples: " + str(total_samples))

        return epoch_losses

    def eval(self, optimizer, criterion, data_loader):
        # accuracy and loss on test set
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_loader:
                outputs, hidden = self(x)
                loss = criterion(outputs, y)
                test_loss += loss.item()
                predicted = torch.argmax(outputs, dim=0)
                total += np.prod(np.array(y.shape))
                correct += (predicted == y).sum().item()
                test_loss /= len(x)
                test_accuracy = correct / total

        return test_loss, test_accuracy
