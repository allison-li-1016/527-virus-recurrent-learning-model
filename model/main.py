import itertools

from loader import CodonLoader
from model import RNNModel
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def main():
    # load codon data for model
    path = "data/resulting-codons.txt"
    codon_loader = CodonLoader(path, test_split=0.2)
    train_data = codon_loader.get_train_data() 
    test_data = codon_loader.get_test_data()

    # creating offset between x and y sequences
    # so that each token is predicting the next token
    train_x, train_y = parse_data(train_data)
    test_x, test_y = parse_data(test_data)

    # Pretraining masking random codons
    # masked_codons = codon_loader.mask(train_data, 0.1)

    # Define hyperparameters
    n_epochs = 100
    lr=0.01

    # Grid search 
    layer_types = ["LSTM", "GRU"]
    hidden_layer_sizes = [128, 256, 512]
    
    for (lt, hls) in itertools.product(layer_types, hidden_layer_sizes):
        # Define model 
        model = RNNModel(lt, len(train_x), len(train_y), len(train_x[0]), hls)
        # Define Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #Train model 
        model.train(optimizer, criterion, n_epochs, train_x, train_y)
        # Validation data
        loss, accuracy, avg_loss= model.eval(model, optimizer, criterion, test_x, test_y)

        # Print stats
        print(str(lt) + " accuracy: " + str(accuracy))
        print(str(lt) + " loss: " + str(loss))
    
        # plotting avg-loss
        plt.plot(avg_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Per-Epoch Losses-" + str(lt))
        plt.savefig("avg-loss-" + str(lt)+".png")
        plt.cla()
        plt.clf()

def parse_data(data):
    # Creating lists that will hold our input and target sequences
    x = []
    y = []

    for i in range(len(data)):
        # Remove last character for input sequence
        x.append(data[i][:-1])
        
        # Remove first character for target sequence
        y.append(data[i][1:])
    
    return x, y

if __name__ == "__main__":
    main()