import itertools

from loader import CodonLoader
from model import RNNModel
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def main():
    # load codon data for model
    path = "../data/resulting-codons.txt"
    codon_loader = CodonLoader(path, num_samples=100, test_split=0.2)
    train_data = codon_loader.get_train_data()
    val_data = codon_loader.get_val_data()
    test_data = codon_loader.get_test_data()

    # Load data from pkl file
    codon_loader.save_encoded_data()

    print(train_data[0])
    # print(data["train_data"][0])

    # creating offset between x and y sequences
    # so that each token is predicting the next token
    train_X, train_y = parse_data(train_data)
    val_X, val_y = parse_data(val_data)
    test_X, test_y = parse_data(test_data)

    # Define hyperparameters
    n_epochs = 100
    lr = 0.01

    # Grid search 
    #layer_types = ["LSTM", "GRU", "RNN"]
    #hidden_layer_sizes = [128, 256, 512]
    layer_types = ["LSTM"]
    hidden_layer_sizes = [1]
    best_params = ()
    best_loss = 1000000000

    for lt, hls in itertools.product(layer_types, hidden_layer_sizes):
        # Define model
        model = RNNModel(lt, len(train_X), len(train_y), len(train_X[0]), hls)
        # Define Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Train model
        model.train(optimizer, criterion, n_epochs, train_X, train_y)
        # Validation data
        loss, accuracy, avg_loss = model.eval(model, optimizer, criterion, n_epochs, test_X, test_y)

        if loss < best_loss:
            best_loss = loss
            best_params = (lt, hls)

        # Print stats
        print(str(lt) + " accuracy: " + str(accuracy))
        print(str(lt) + " loss: " + str(loss))
    
        # plotting loss per epoch
        plt.plot(avg_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Per-Epoch Losses-" + str(lt))
        plt.savefig("avg-loss-" + str(lt) + ".png")
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
