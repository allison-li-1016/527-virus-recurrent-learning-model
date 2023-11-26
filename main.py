import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from loader.loader import CodonLoader
from model.autoreg_model import AutoregressiveRNNModel
from model.masking_model import MaskedRNNModel
import os

EPOCHS = 100
VERBOSE = True


def main():
    path = "data/resulting-codons.txt"
    codon_loader = CodonLoader(
        path,
        num_samples=20,
        batch_size=32,
        num_epochs=EPOCHS,
        offset=True,
        test_split=0.2,
    )

    masked_codon_loader = CodonLoader(
        path,
        num_samples=20,
        batch_size=32,
        num_epochs=EPOCHS,
        offset=False,
        test_split=0.2,
    )

    model_types = [("MASKED", masked_codon_loader), ("AUTOREGRESSIVE", codon_loader)]

    for m in model_types:
        loader = m[1]
        train_loader, val_loader, test_loader = loader.data_loader()
        evaluate(m[0], train_loader, val_loader, test_loader)


def evaluate(model_type, train_loader, val_loader, test_loader):
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
        # TODO: not sure how not to hard code number of features values here
        if model_type == "MASKED":
            model = MaskedRNNModel(
                layer_type=lt, input_size=64, output_size=64, hidden_dim=hls, n_layers=1
            )
        else:
            model = AutoregressiveRNNModel(
                layer_type=lt, input_size=64, output_size=64, hidden_dim=hls, n_layers=1
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print("model type: " + model_type)
        print("layer type: " + lt)
        print("num hidden layers: " + str(hls))

        train_loss = model.train(
            optimizer, criterion, EPOCHS, train_loader, verbose=VERBOSE
        )
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

        #TODO: val loss is now a single value and not a list since there is no training, can't rly plot it against training loss
        plot_loss(train_loss, val_loss, lt, hls, model_type)

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

    plot_loss(best_train_loss, test_loss, best_model_name, best_num_layers, model_type, best=True)


def plot_loss(train_loss, val_loss, model_name, num_layers, model_type, best=False):
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Per-Epoch Losses-" + str(model_name) + "-" + model_type)
    plt.legend()
    # Creating file path
    if best:
        filename = f"images/{model_type.lower()}/avg-loss-" + str(model_name)+ "-"+ str(num_layers)+ "-best-" + ".png"
    else:
        filename = f"images/{model_type.lower()}/avg-loss-" + str(model_name)+ "-"+ str(num_layers)+ "-" + model_type + ".png"
    # Extract the directory path from the filename
    directory = os.path.dirname(filename)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(filename)
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    main()
