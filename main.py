import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from loader.loader import CodonLoader
from model.autoreg_model import AutoregressiveRNNModel
from model.masking_model import MaskedRNNModel
import os

EPOCHS = 10
VERBOSE = True
LOAD_BEST = False


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
        num_samples=50,
        batch_size=32,
        num_epochs=EPOCHS,
        offset=False,
        test_split=0.2,
    )

    model_types = [("MASKED", masked_codon_loader), ("AUTOREGRESSIVE", codon_loader)]

    for m in model_types:
        model_type = m[0]
        loader = m[1]
        if not LOAD_BEST:
            evaluate(model_type, loader)
        else:
            load(model_type, loader)


def load(model_type, loader):
    train_loader, val_loader, test_loader = loader.data_loader()
    path = f"model_files/{model_type.lower()}/best-model.pt"
    #TODO: not sure how to not hard code parameters
    if model_type == "MASKED":
            model = MaskedRNNModel(
                layer_type="RNN", input_size=64, output_size=64, hidden_dim=32, n_layers=1
            )
            criterion = nn.CrossEntropyLoss()
    else:
            model = AutoregressiveRNNModel(
                layer_type="RNN", input_size=64, output_size=64, hidden_dim=32, n_layers=1
            )
    model.load_state_dict(torch.load(path))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, _ = model.eval(criterion, test_loader, predict=True)
    print(f"model: {model_type}")
    print( f"test loss: {test_loss:.4f} | test acc: {test_accuracy:.4f}")

def evaluate(model_type, loader):
    train_loader, val_loader, test_loader = loader.data_loader()

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
    best_val_loss = []

    for lt, hls in itertools.product(layer_types, hidden_layer_sizes):
        # TODO: not sure how not to hard code number of features values here
        if model_type == "MASKED":
            model = MaskedRNNModel(
                layer_type=lt, input_size=64, output_size=64, hidden_dim=hls, n_layers=1
            )
            criterion = nn.CrossEntropyLoss()
        else:
            model = AutoregressiveRNNModel(
                layer_type=lt, input_size=64, output_size=64, hidden_dim=hls, n_layers=1
            )
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        print("model type: " + model_type)
        print("layer type: " + lt)
        print("num hidden layers: " + str(hls))

        training_losses = []
        validation_losses = []

        for epoch in range(EPOCHS):
            epoch_resolution = max(1, EPOCHS // 10)
            # train the model
            train_loss, train_accuracy, train_total_samples = model.train(
                optimizer, criterion, train_loader)
            
            
            # validate the model
            val_loss, val_accuracy, val_total_samples = model.eval(criterion, val_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = model
                best_model_name = lt
                best_num_layers = hls
                best_train_loss = training_losses
                best_val_loss = validation_losses
            
            #printing stats
            print("Epoch: {}/{}.............".format(epoch, EPOCHS), end=" ")
            print()
            if VERBOSE and (epoch % epoch_resolution == 0):
                print(
                    "Train Loss: {:.4f}, Train Accuracy: {:.2f}%".format(
                        train_loss, train_accuracy * 100
                    )
                )
                print("training correct labels: " + str(train_accuracy*train_total_samples))
                print("training total samples: " + str(train_total_samples))

                print(
                    "Val Loss: {:.4f}, Val Accuracy: {:.2f}%".format(
                        val_loss, val_accuracy * 100
                    )
                )
                print("validation correct labels: " + str(val_accuracy*val_total_samples))
                print("validation total samples: " + str(val_total_samples))
                print()

            print("Summary................")
            print(
                 f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_accuracy:.4f}"
            )
            print()

            training_losses.append(train_loss)
            validation_losses.append(val_loss)

        plot_loss(training_losses, validation_losses, lt, hls, model_type)

    model = best_params
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, _ = model.eval(criterion, test_loader, predict=True)

    # for x, y in test_loader:
    #     print("actual:")
    #     print(loader.decode_codons(y[0]))
    #     print("predicted:")
    #     print(loader.decode_codons(model.predict(x)[0]))

    print(
        "Best model is : "
        + str(best_model_name)
        + " with hidden layer size "
        + str(best_num_layers)
    )
    print("test accuracy: " + str(test_accuracy * 100) + "%")
    print("test loss: " + str(test_loss))

    plot_loss(
        best_train_loss,
        best_val_loss,
        best_model_name,
        best_num_layers,
        model_type,
        best=True,
    )

    # save trained best modell
    path = f"model_files/{model_type.lower()}/best-model.pt"
    # Extract the directory path from the filename
    directory = os.path.dirname(path)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), path)


def plot_loss(train_loss, val_loss, model_name, num_layers, model_type, best=False):
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Per-Epoch Losses-" + str(model_name) + "-" + model_type)
    plt.legend()
    # Creating file path
    if best:
        filename = (
            f"images/{model_type.lower()}/avg-loss-"
            + str(model_name)
            + "-"
            + str(num_layers)
            + "-best-"
            + ".png"
        )
    else:
        filename = (
            f"images/{model_type.lower()}/avg-loss-"
            + str(model_name)
            + "-"
            + str(num_layers)
            + "-"
            + model_type
            + ".png"
        )
    # Extract the directory path from the filename
    directory = os.path.dirname(filename)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(filename)
    plt.cla()
    plt.clf()


def custom_masked_loss(outputs, targets, mask_value=-1):
    # Mask out the loss for positions where the input was masked
    mask = targets != mask_value
    masked_outputs = outputs[mask]
    masked_targets = targets[mask]
    return nn.functional.cross_entropy(masked_outputs, masked_targets)


if __name__ == "__main__":
    main()
