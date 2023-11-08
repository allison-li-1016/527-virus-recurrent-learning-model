import random
import itertools
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class CodonLoader:
    def __init__(self, file_path, num_samples=10000, test_split=0.2, random_state=42):
        self.file_path = file_path
        self.test_split = test_split
        self.random_state = random_state
        self.num_samples = num_samples
        self.save_path = file_path.replace(".txt", ".pkl")
        self.codon_data = self._load_data()
        self.train_data, self.val_data, self.test_data = self._split_data()

    def _load_data(self):
        with open(self.file_path, "r") as file:
            lines = file.readlines()
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            codon_data = [cleaned_lines[i + 1] for i in range(0, len(cleaned_lines), 2)]
            # Get num_samples codons
            codon_data = codon_data[: self.num_samples]
            # Remove spaces from codon data
            codon_data = [codon.replace(" ", "") for codon in codon_data]
            # Split codon data into lists
            codon_data = [codon.split(",") for codon in codon_data]
            # Remove empty codon at the end of each list
            codon_data = [codon[:-1] for codon in codon_data]
            # Remove start codon at the beginning of each list
            codon_data = [codon[1:] for codon in codon_data]
            # Replace any codon with N in it with NNN
            codon_data = [
                [codon if "N" not in codon else "NNN" for codon in codon_sequence]
                for codon_sequence in codon_data
            ]
            # Encode codons as one-hot vectors
            codon_data = self._encode_one_hot(codon_data)
        return codon_data

    def _split_data(self):
        X = self.codon_data
        # Creating a dummy target variable
        y = [None] * len(X)
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=self.test_split, random_state=self.random_state
        )
        # Split training data into training and validation sets
        y = [None] * len(X_train)
        X_train, X_val, _, _ = train_test_split(
            X_train, y, test_size=self.test_split, random_state=self.random_state
        )
        return X_train, X_val, X_test

    def _encode_one_hot(self, codon_list):
        nucleotides = ["A", "C", "G", "T"]
        # Encode codons as one-hot vectors
        possible_codons = ["".join(x) for x in list(itertools.product(nucleotides, repeat=3))]
        encoder = OneHotEncoder(categories=[possible_codons], handle_unknown="ignore")
        # Flatten data to train encoder
        flat_codon_list = [codon for codon_sequence in codon_list for codon in codon_sequence]
        encoder.fit([[codon] for codon in flat_codon_list])
        encoded_codons = []
        for codon_sequence in codon_list:
            encoded_codon_sequence = []
            for codon in codon_sequence:
                encoded_codon = encoder.transform([[codon]]).toarray()
                encoded_codon_sequence.append(encoded_codon)
            encoded_codons.append(encoded_codon_sequence)

        return encoded_codons

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data

    def save_encoded_data(self):
        with open(self.save_path, "wb") as file:
            encoded_data = {
                "train_data": self.train_data,
                "val_data": self.val_data,
                "test_data": self.test_data,
            }
            pickle.dump(encoded_data, file)

    @staticmethod
    def load_encoded_data(load_path):
        with open(load_path, "rb") as file:
            encoded_data = pickle.load(file)
            # self.train_data = encoded_data["train_data"]
            # self.val_data = encoded_data["val_data"]
            # self.test_data = encoded_data["test_data"]
            return encoded_data

    @staticmethod
    def random_mask(codon_list, mask_percentage):
        masked_codons = []
        for codon_sequence in codon_list:
            masked_codon = []
            for codon in codon_sequence:
                if random.random() < mask_percentage:
                    masked_codon.append("NNN")
                else:
                    masked_codon.append(codon)
            masked_codons.append(masked_codon)
        return masked_codons


# file_path = "../data/resulting-codons.txt"
# codon_loader = CodonLoader(file_path, test_split=0.2)
# train_data = codon_loader.get_train_data()
# test_data = codon_loader.get_test_data()
