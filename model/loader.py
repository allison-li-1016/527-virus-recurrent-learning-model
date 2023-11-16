import random
import itertools
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class CodonDataset(Dataset):
    def __init__(self, file_path, num_samples=None, random_state=42):
        self.file_path = file_path
        self.random_state = random_state
        self.num_samples = num_samples
        self.codon_data, self.y = self._load_data()

    def __len__(self):
        return len(self.codon_data)

    def __getitem__(self, idx):
        return torch.tensor(
            np.array(self.codon_data[idx]), dtype=torch.float32
        ), torch.tensor(np.array(self.y[idx]), dtype=torch.float32)

    def _load_data(self):
        with open(self.file_path, "r") as file:
            lines = file.readlines()
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            codon_data = [cleaned_lines[i + 1] for i in range(0, len(cleaned_lines), 2)]
            codon_data = [
                codon.replace(" ", "").split(",")[:-1][1:] for codon in codon_data
            ]
            codon_data = [
                [codon if "N" not in codon else "NNN" for codon in codon_sequence]
                for codon_sequence in codon_data
            ]

            random.shuffle(codon_data)
            if self.num_samples is not None:
                codon_data = codon_data[: self.num_samples]

            nucleotides = ["A", "C", "G", "T"]
            possible_codons = [
                "".join(x) for x in list(itertools.product(nucleotides, repeat=3))
            ]
            encoder = OneHotEncoder(
                categories=[possible_codons], handle_unknown="ignore"
            )

            flat_codon_list = [
                codon for codon_sequence in codon_data for codon in codon_sequence
            ]
            encoder.fit([[codon] for codon in flat_codon_list])

            encoded_codons = [
                [encoder.transform([[codon]]).toarray()[0] for codon in codon_sequence]
                for codon_sequence in codon_data
            ]

            # Get offset data
            codon_data_offset = [sequence[:-1] for sequence in encoded_codons]
            y_offset = [sequence[1:] for sequence in encoded_codons]

        return codon_data_offset, y_offset

    def decode(encoded_sequences):
        # Reverse the one-hot encoding and convert the data back to its original form
        nucleotides = ["A", "C", "G", "T"]
        possible_codons = ["".join(x) for x in list(itertools.product(nucleotides, repeat=3))]
        codon_dict = {}
        for i, codon in enumerate(possible_codons):
            encoded_codon = np.zeros(len(possible_codons))
            encoded_codon[i] = 1
            codon_dict[tuple(encoded_codon)] = codon

        #for codon in encoded_sequences[1]:
        encoded_tuple = tuple(tuple(inner_list) for inner_list in encoded_sequences)
        decoded_codons = codon_dict.get(encoded_tuple, "NNN")
    
        return decoded_codons


class CodonLoader:
    def __init__(
        self,
        file_path,
        num_samples=None,
        batch_size=32,
        num_epochs=10,
        test_split=0.2,
        random_state=42,
    ):
        self.file_path = file_path
        self.test_split = test_split
        self.random_state = random_state
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_path = file_path.replace(".txt", ".pkl")

    def data_loader(self):
        dataset = CodonDataset(
            self.file_path, num_samples=self.num_samples, random_state=self.random_state
        )
        train_size = int((1 - self.test_split) * len(dataset))
        val_size = int(
            self.test_split * train_size
        )  # Use a fraction of the training set for validation
        train_size -= val_size  # Adjust the training set size
        test_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader

