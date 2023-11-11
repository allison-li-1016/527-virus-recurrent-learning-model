import itertools

from loader import CodonLoader
from model import RNNModel
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def main():
    # load codon data for model
    path = "../data/resulting-codons.txt"
    codon_loader = CodonLoader(path, num_samples=1000, test_split=0.2)
    # creating offset between x and y sequences
    # so that each token is predicting the next token
    train_data = parse_data(codon_loader.get_train_data())
    val_data = parse_data(codon_loader.get_val_data())
    test_data = parse_data(codon_loader.get_test_data())

    # Load data from pkl file
    # codon_loader.save_encoded_data()


    # Define hyperparameters
    n_epochs = 100
    lr = 0.01

    # Grid search
    layer_types = ["RNN", "LSTM", "GRU"]
    hidden_layer_sizes = [128, 256, 512]
    #layer_types = ["RNN"]
    #hidden_layer_sizes = [64]
    best_params = []
    best_model_name = ""
    best_num_layers = 0
    best_loss = 1000000000

    for lt, hls in itertools.product(layer_types, hidden_layer_sizes):
        # Define model

        # not sure how not to hard code number of features values here
        model = RNNModel(lt, 64, 64, hls, 1)

        # Define Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Train model

        print("layer type: " + lt)
        print("num hidden layers: " + str(hls))
        model.train(optimizer, criterion, n_epochs, train_data)
        # Validation data
        loss, accuracy, avg_loss = model.eval(optimizer, criterion, n_epochs, val_data)

        if loss < best_loss:
            best_loss = loss
            best_params = (model)
            best_model_name = lt
            best_num_layers = hls

        # Print stats
        print(str(lt) + " accuracy: " + str(accuracy * 100) + "%")
        print(str(lt) + " loss: " + str(loss))

        # plotting loss per epoch
        plt.plot(avg_loss)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Per-Epoch Losses-" + str(lt))
        plt.savefig("avg-loss-" + str(lt) +  "-" + str(hls) + ".png")
        plt.cla()
        plt.clf()
    
    model = best_params
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss, accuracy, avg_loss = model.eval(optimizer, criterion, n_epochs, test_data)
    print("Best model is : " + str(best_model_name) + " with hidden layer size " + str(best_num_layers))
    print("test accuracy: " + str(accuracy * 100) + "%")
    print("test loss: " + str(loss))
    # plotting loss per epoch
    plt.plot(avg_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Per-Epoch Losses-" + str(best_model_name))
    plt.savefig("avg-loss-" + str(best_model_name) +  "-" + str(best_num_layers) + ".png")
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
    
    # change lists to tensors
    x = np.array(x)
    y = np.array(y)
    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    return DataLoader(TensorDataset(x,y), 32, shuffle=True)


if __name__ == "__main__":
    main()
