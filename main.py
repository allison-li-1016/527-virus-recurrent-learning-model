import itertools
import umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn as nn

from loader.loader import CodonLoader
from model.autoreg_model import AutoregressiveRNNModel
from model.masking_model import MaskedRNNModel
import os

EPOCHS = 5
VERBOSE = True
LOAD_BEST = True


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
        model_type = m[0]
        loader = m[1]
        if not LOAD_BEST:
            evaluate(model_type, loader)
        else:
            load(model_type, loader)
        break


def load(model_type, loader):
    train_loader, val_loader, test_loader = loader.data_loader()
    path = f"model_files/{model_type.lower()}/best-model.pt"
    # TODO: not sure how to not hard code parameters
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
    test_loss, test_accuracy, total, probabilities = model.eval(criterion, test_loader)
    print(f"model: {model_type}")
    print(f"test loss: {test_loss:.4f} | test acc: {test_accuracy:.4f}")

    """Plot UMAP of codon probabilities""" ""
    flattened_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
    true_labels = torch.tensor([])
    for x, y in test_loader:
        true_labels = torch.cat((true_labels, y), dim=0)

    flattened_labels = true_labels.reshape(-1, true_labels.shape[-1])
    flattened_labels = torch.argmax(flattened_labels, dim=1)

    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric="correlation")
    umap_embeddings = umap_model.fit_transform(flattened_probabilities)

    codon_mapping = loader.encoder.get_feature_names_out()
    codon_mapping = [codon[-3:] for codon in codon_mapping]

    _, label_indices = torch.unique(flattened_labels, return_inverse=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=flattened_labels,
        cmap="Spectral",
    )

    legend_handles = [
        Patch(color=scatter.cmap(scatter.norm(label)), label=f"{codon_mapping[label]}")
        for label in np.unique(label_indices)
    ]

    plt.legend(
        handles=legend_handles,
        title="Codon Names",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        ncol=4,
    )
    plt.tight_layout()
    plt.savefig("images/umap_codons.png")

    """Plot UMAP of amino acids"""
    codon_to_amino_acid_mapping = {
        "AAA": "Lys",
        "AAG": "Lys",
        "AAC": "Asn",
        "AAT": "Asn",
        "ACA": "Thr",
        "ACC": "Thr",
        "ACG": "Thr",
        "ACT": "Thr",
        "AGA": "Arg",
        "AGC": "Ser",
        "AGG": "Arg",
        "AGT": "Ser",
        "ATA": "Ile",
        "ATC": "Ile",
        "ATG": "Met",
        "ATT": "Ile",
        "CAA": "Gln",
        "CAG": "Gln",
        "CAC": "His",
        "CAT": "His",
        "CCA": "Pro",
        "CCC": "Pro",
        "CCG": "Pro",
        "CCT": "Pro",
        "CGA": "Arg",
        "CGC": "Arg",
        "CGG": "Arg",
        "CGT": "Arg",
        "CTA": "Leu",
        "CTC": "Leu",
        "CTG": "Leu",
        "CTT": "Leu",
        "GAA": "Glu",
        "GAG": "Glu",
        "GAC": "Asp",
        "GAT": "Asp",
        "GCA": "Ala",
        "GCC": "Ala",
        "GCG": "Ala",
        "GCT": "Ala",
        "GGA": "Gly",
        "GGC": "Gly",
        "GGG": "Gly",
        "GGT": "Gly",
        "GTA": "Val",
        "GTC": "Val",
        "GTG": "Val",
        "GTT": "Val",
        "TAA": "STOP",
        "TAG": "STOP",
        "TAC": "Tyr",
        "TAT": "Tyr",
        "TCA": "Ser",
        "TCC": "Ser",
        "TCG": "Ser",
        "TCT": "Ser",
        "TGA": "STOP",
        "TGC": "Cys",
        "TGG": "Trp",
        "TGT": "Cys",
        "TTA": "Leu",
        "TTC": "Phe",
        "TTG": "Leu",
        "TTT": "Phe",
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    flattened_labels = [codon_mapping[codon] for codon in flattened_labels]
    amino_acids = [codon_to_amino_acid_mapping[codon] for codon in flattened_labels]
    amino_acid_to_index = {
        amino_acid: i for i, amino_acid in enumerate(set(amino_acids))
    }
    indexed_amino_acids = [
        amino_acid_to_index[amino_acid] for amino_acid in amino_acids
    ]

    scatter = plt.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=indexed_amino_acids,
        cmap="Spectral",
    )

    _, label_indices = torch.unique(
        torch.tensor(indexed_amino_acids), return_inverse=True
    )
    legend_handles = [
        Patch(
            color=scatter.cmap(scatter.norm(label)),
            label=f"{list(amino_acid_to_index.keys())[label]}",
        )
        for label in np.unique(label_indices)
    ]

    plt.legend(
        handles=legend_handles,
        title="Amino Acids",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        ncol=2,
    )
    plt.tight_layout()

    plt.savefig("images/umap_amino_acids.png")


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
                optimizer, criterion, train_loader
            )

            # validate the model
            val_loss, val_accuracy, val_total_samples, _ = model.eval(
                criterion, val_loader
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = model
                best_model_name = lt
                best_num_layers = hls
                best_train_loss = training_losses
                best_val_loss = validation_losses

            # printing stats
            print("Epoch: {}/{}.............".format(epoch, EPOCHS), end=" ")
            print()
            if VERBOSE and (epoch % epoch_resolution == 0):
                print(
                    "Train Loss: {:.4f}, Train Accuracy: {:.2f}%".format(
                        train_loss, train_accuracy * 100
                    )
                )
                print(
                    "training correct labels: "
                    + str(train_accuracy * train_total_samples)
                )
                print("training total samples: " + str(train_total_samples))

                print(
                    "Val Loss: {:.4f}, Val Accuracy: {:.2f}%".format(
                        val_loss, val_accuracy * 100
                    )
                )
                print(
                    "validation correct labels: "
                    + str(val_accuracy * val_total_samples)
                )
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
    test_loss, test_accuracy, _, probabilities = model.eval(criterion, test_loader)

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

    # Save best model
    path = f"model_files/{model_type.lower()}/best-model.pt"
    directory = os.path.dirname(path)
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
