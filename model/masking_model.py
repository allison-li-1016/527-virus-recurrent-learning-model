import torch
import torch.nn as nn

import numpy as np
import math
from loader.loader import CodonDataset, CodonLoader


class MaskedRNNModel(nn.Module):
    def __init__(
        self, layer_type, input_size, output_size, hidden_dim, n_layers, mask_prob=0.0
    ):
        super(MaskedRNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.layer_type = layer_type
        self.mask_prob = mask_prob

        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.activation = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.size(0)

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        unseq_mask = torch.sum(x, dim=2) == 0
        unsequenced_inputs = x.masked_fill(unseq_mask, 0)

        mask = torch.rand(unsequenced_inputs.size()) < self.mask_prob
        masked_inputs = unsequenced_inputs.masked_fill(mask, -1)

        # mask = torch.rand(x.size()) < self.mask_prob
        # masked_inputs = x.masked_fill(mask, -1)

        out, hidden_n = self.rnn(masked_inputs, hidden)
        out = self.activation(out)
        out = self.fc(out)

        out = self.softmax(out)

        return out, hidden_n, mask, unseq_mask

    def predict(self, x):
        with torch.no_grad():
            logits, hidden, mask, unseq_mask = self(x)
            _, predicted = torch.max(logits, dim=2)
            predicted_one_hot = nn.functional.one_hot(predicted, num_classes=64)
            return predicted_one_hot

    def train(self, optimizer, criterion, epochs, train_loader, verbose=True):
        epoch_losses = []
        epoch_resolution = max(1, epochs // 10)

        for epoch in range(1, epochs + 1):
            epoch_loss = 0
            num_batches = 0
            for x, y in train_loader:
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                output, hidden, mask, unseq_mask = self(x)
                print(output.size())
                masked_y = y.masked_fill(unseq_mask, 0)
                loss = criterion(output, masked_y)
                loss.backward()  # Does backpropagation and calculates gradients
                epoch_loss += loss.item()
                optimizer.step()  # Updates the weights accordingly

                # Calculate accuracy
                predicted_labels = torch.argmax(output, dim=0)
                correct_predictions = (predicted_labels == y).sum().item()
                total_samples = np.prod(np.array(y.shape))
                accuracy = correct_predictions / total_samples
                # Calculate num_batches
                num_batches += 1

            print("Number of batches: " + str(num_batches))
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

    def eval(self, criterion, data_loader, predict=False):
        # accuracy and loss on test set
        test_loss = 0.0
        correct = 0
        total = 0
        test_accuracy = 0
        with torch.no_grad():
            for x, y in data_loader:
                outputs, hidden, mask, unseq_mask = self(x)
                masked_y = y.masked_fill(unseq_mask, -1)
                loss = criterion(outputs, masked_y=y.masked_fill(unseq_mask, 0))
                test_loss += loss.item()
                predicted = self.predict(x)
                total += np.prod(np.array(y.shape))
                correct += (predicted == y).sum().item()
                test_loss /= len(x)
                test_accuracy = correct / total

        return test_loss, test_accuracy

    def masked_entropy_loss(self, x, y, mask):
        # Obtaining dimensions of input matrix
        batch_size, _, codon_dict_length = x.shape

        # Use negative log likelihood loss (cross-entropy) with reduction='none'
        loss = nn.CrossEntropyLoss(x, y, reduction="none")

        # Flatten along the second dimension
        mask2d = mask.view(batch_size, -1, codon_dict_length)

        # Apply the mask to ignore the loss for unmasked positions
        masked_loss = loss * mask2d

        # Calculate the mean loss only for masked positions
        masked_positions = torch.sum(mask2d)

        if masked_positions == 0:
            # Handle the case where there are no non-masked positions to avoid division by zero
            return torch.tensor(0.0, requires_grad=True)
        else:
            return torch.sum(masked_loss) / masked_positions
