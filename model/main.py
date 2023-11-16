import itertools

from loader import CodonLoader
from model import RNNModel
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

EPOCHS = 100
VERBOSE = True


def main():
    # load codon data for model
    path = "../data/resulting-codons.txt"
    codon_loader = CodonLoader(
        path, num_samples=10, batch_size=32, num_epochs=EPOCHS, test_split=0.2
    )
    train_loader, val_loader, test_loader = codon_loader.data_loader()

    # Define hyperparameters
    lr = 0.01

    # Grid search
    layer_types = ["RNN"]
    hidden_layer_sizes = [32]

    best_params = []
    best_model_name = ""
    best_num_layers = 0
    best_loss = np.inf
    best_train_loss = []

    for lt, hls in itertools.product(layer_types, hidden_layer_sizes):
        # Define model

        # not sure how not to hard code number of features values here
        model = RNNModel(
            layer_type=lt, input_size=64, output_size=64, hidden_dim=hls, n_layers=1
        )

        # Define Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Train model
        print("layer type: " + lt)
        print("num hidden layers: " + str(hls))

        train_loss = model.train(
            optimizer, criterion, EPOCHS, train_loader, verbose=VERBOSE
        )
        # Validation data
        val_loss, accuracy = model.eval(criterion, val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_params = model
            best_model_name = lt
            best_num_layers = hls
            best_train_loss = train_loss

        print(
            f"train loss: {train_loss[-1]:.4f} | val loss: {val_loss:.4f} | val acc: {accuracy:.4f}"
        )

        # plotting loss per epoch
        plt.plot(train_loss, label="train loss")
        plt.plot(val_loss, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Per-Epoch Losses-" + str(lt))
        plt.legend()
        plt.savefig("avg-loss-" + str(lt) + "-" + str(hls) + ".png")
        plt.cla()
        plt.clf()

    model = best_params
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    test_loss, accuracy = model.eval(criterion, test_loader, predict=True)
    print(
        "Best model is : "
        + str(best_model_name)
        + " with hidden layer size "
        + str(best_num_layers)
    )
    print("test accuracy: " + str(accuracy * 100) + "%")
    print("test loss: " + str(test_loss))
    # plotting loss per epoch
    plt.plot(best_train_loss, label="train loss")
    plt.plot(test_loss, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Per-Epoch Losses-" + str(best_model_name))
    plt.savefig(
        "avg-loss-" + str(best_model_name) + "-" + str(best_num_layers) + ".png"
    )
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    main()
