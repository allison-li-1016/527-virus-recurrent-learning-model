import random
import itertools
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class CodonDataset(Dataset):
    def __init__(
        self,
        file_path,
        num_samples=None,
        offset=True,
        encoder=OneHotEncoder(),
        random_state=42,
    ):
        self.file_path = file_path
        self.random_state = random_state
        self.num_samples = num_samples
        self.offset = offset
        self.encoder = encoder
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

            codon_data = [
                codon_sequence
                for codon_sequence in codon_data
                if "NNN" not in codon_sequence
            ]

            random.shuffle(codon_data)
            if self.num_samples is not None:
                codon_data = codon_data[: self.num_samples]

            flat_codon_list = [
                codon for codon_sequence in codon_data for codon in codon_sequence
            ]
            self.encoder.fit([[codon] for codon in flat_codon_list])

            encoded_codons = [
                [
                    self.encoder.transform([[codon]]).toarray()[0]
                    for codon in codon_sequence
                ]
                for codon_sequence in codon_data
            ]

            # Get offset data
            if self.offset:
                codon_data_offset = [sequence[:-1] for sequence in encoded_codons]
                y_offset = [sequence[1:] for sequence in encoded_codons]
                return codon_data_offset, y_offset
            # Get non-offset data and mask some percent of the codons
            else:
                codon_data_masked = [
                    [
                        codon if random.random() > 0.15 else np.zeros(64)
                        for codon in codon_sequence
                    ]
                    for codon_sequence in encoded_codons
                ]
                return codon_data_masked, encoded_codons


class CodonLoader:
    def __init__(
        self,
        file_path,
        num_samples=None,
        batch_size=32,
        num_epochs=10,
        offset=True,
        test_split=0.2,
        random_state=42,
    ):
        self.file_path = file_path
        self.test_split = test_split
        self.random_state = random_state
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.offset = offset
        self.save_path = file_path.replace(".txt", ".pkl")

        self.nucleotides = ["A", "C", "G", "T"]
        self.possible_codons = [
            "".join(x) for x in list(itertools.product(self.nucleotides, repeat=3))
        ]

        self.encoder = OneHotEncoder(
            categories=[self.possible_codons], handle_unknown="ignore"
        )

    def data_loader(self):
        dataset = CodonDataset(
            self.file_path,
            num_samples=self.num_samples,
            offset=self.offset,
            encoder=self.encoder,
            random_state=self.random_state,
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

    def decode_codons(self, encoded_sequences):
        decoded_codons = self.encoder.inverse_transform(encoded_sequences)
        decoded_codons = [
            ["MASK" if codon == None else codon for codon in sequence]
            for sequence in decoded_codons
        ]
        return decoded_codons
