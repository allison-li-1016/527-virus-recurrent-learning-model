import torch
import torch.nn as nn

import numpy as np
import math
from loader.loader import CodonDataset


class MaskedRNNModel(nn.Module):
    def __init__(
        self, layer_type, input_size, output_size, hidden_dim, n_layers, mask_prob=0.0
    ):
        super(MaskedRNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.layer_type = layer_type
        self.mask_prob = mask_prob

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

        #TODO: Double check if '0' is an appropriate value to mask with
        #masked_inputs = self.mask_tensor(x)
        masked_inputs = x

        # Special case for LSTM
        if self.layer_type == "LSTM":
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
            hidden = (hidden, c0)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden_n = self.rnn(masked_inputs, hidden)

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

    def eval(self, criterion, data_loader, predict=False):
        # accuracy and loss on test set
        test_loss = 0.0
        correct = 0
        total = 0
        test_accuracy = 0
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

                # predicting output
                if predict:
                    #TODO: the dimensions of this might be incorrect. Check if predicted needs to be transposed or not
                    #TODO: expected dimension of decode is (batch size, seq length, codon dict length)
                    print("predicted sequence: ")
                    print(*CodonDataset.decode(predicted.numpy().transpose()), sep=", ")

        return test_loss, test_accuracy
    
    def masked_entropy_loss(self,x, y, mask):
        # Obtaining dimensions of input matrix
        batch_size, _, codon_dict_length = x.shape

        # Use negative log likelihood loss (cross-entropy) with reduction='none'
        loss = nn.CrossEntropyLoss(x,y,reduction='none')

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

    def mask_tensor(self, matrix):
        # Ensure mask_percentage is a valid percentage value
        if self.mask_prob < 0 or self.mask_prob > 1:
            raise ValueError("Mask percentage should be between 0 and 1.")

        # Calculate the number of codons to be masked per batch
        batch_size, sequence_length, codon_length = matrix.shape
        num_codons_to_mask = math.ceil(self.mask_prob * (sequence_length))

        #print(num_codons_to_mask)

        # Create a flat mask with zeros (masked) and ones (not masked)
        matrices = []

         # Randomly choose codons to be masked for each batch
        for batch in range(batch_size):
            m = np.zeros((sequence_length, codon_length))
            masked_codon_indices = np.random.choice(sequence_length, num_codons_to_mask)
            m[masked_codon_indices, :] = 1
            matrices.append(m)

        # stack matrices for each batch into 1 matrix
        mask = np.stack(matrices)
        
        # Apply the mask to the original matrix
        masked_matrix = np.multiply(matrix, 1 - mask)

        # Convert matrix back to tensor of floats
        masked_matrix = masked_matrix.float()


        #TODO: masking around ~1000 codons whereas only supposed to mask 191 * 4. could be do to unsequenced codons. Consistently getting 197 masked codons even with percentage of 0

        #masked_decode = CodonDataset.decode(masked_matrix.numpy())
        #print("masked sequence: ")
        #print(*CodonDataset.decode(masked_matrix.numpy()), sep=", ")
        #print(np.column_stack(np.where(masked_decode == 'MASKED')).shape)

        return masked_matrix
