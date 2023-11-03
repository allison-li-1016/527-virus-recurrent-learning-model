import torch
import torch.nn as nn
# Seq to seq model
class RNNModel:
    #would be nice to be able to pass in RNN type as parameter too
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        #super simple LSTM model 
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        #defining layers
        self.rnn = nn.LSTM(input_size, hidden_dim, n_layers)  
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
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        # return output layer and hidden state (for training and RNN optimization)
        return out, hidden 

