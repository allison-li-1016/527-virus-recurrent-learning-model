import torch
import torch.nn as nn

# Seq to seq model
class RNNModel:
    #would be nice to be able to pass in RNN type as parameter too
    def __init__(self, layer_type, input_size, output_size, hidden_dim, n_layers):
        #super simple LSTM model 
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #defining layers
        self.rnn = layer_type(input_size, hidden_dim, n_layers)  
        #fully connected layer -> connect network to output labels
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        #Initializing hidden state for first input 
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # -1 is n of original output tensor, second dimension is the same as the hidden layer

        #this might need to be deleted if its many to many
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        # return output layer and hidden state (for training and RNN optimization)
        return out, hidden 

    def train(self,optimizer, criterion, epochs,x,y):
        epoch_losses = []
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            output, hidden = self.forward(x)
            loss = criterion(output, y.view(-1).long())
            loss.backward() # Does backpropagation and calculates gradients
            epoch_losses.append(loss.item())
            optimizer.step() # Updates the weights accordingly

            # Calculate accuracy
            predicted_labels = torch.argmax(output, dim=1)
            correct_predictions = (predicted_labels == y).sum().item()
            total_samples = y.size(0)
            accuracy = correct_predictions / total_samples
            
            if epoch%10 == 0:
                print('Epoch: {}/{}.............'.format(epoch, epochs), end=' ')
                print("Loss: {:.4f}, Accuracy: {:.2f}%".format(loss.item(), accuracy * 100))
        
        return epoch_losses
    
    def eval(self,optimizer, criterion, epochs, x, y):
        # train and get per epoch losses
        avg_loss = self.train(optimizer, criterion, epochs, x, y)
        # accuracy and loss on test set
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in zip(x,y):
                outputs = self.forward(x_val)
                loss = criterion(x_val,y_val)
                test_loss += loss.item()
                _, predicted = torch.argmax(outputs, dim=1)
                total += y.size(0)
                correct += (predicted == y_val).sum().item()
            test_loss /= len(x)
            test_accuracy = correct / total

        return test_loss, test_accuracy, avg_loss

