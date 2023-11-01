import random
from sklearn.model_selection import train_test_split


class CodonLoader:
    def __init__(self, file_path, test_split=0.2, random_state=42):
        self.file_path = file_path
        self.test_split = test_split
        self.random_state = random_state
        self.codon_data = self._load_data()
        self.train_data, self.test_data = self._split_data()

    def _load_data(self):
        with open(self.file_path, "r") as file:
            lines = file.readlines()
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            codon_data = [cleaned_lines[i + 1] for i in range(0, len(cleaned_lines), 2)]
            # Remove spaces from codon data
            codon_data = [codon.replace(" ", "") for codon in codon_data]
        return codon_data

    def _split_data(self):
        X = self.codon_data
        # Creating a dummy target variable
        y = [None] * len(X)
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=self.test_split, random_state=self.random_state
        )
        return X_train, X_test

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data


file_path = "../data/resulting-codons.txt"
codon_loader = CodonLoader(file_path, test_split=0.2)
train_data = codon_loader.get_train_data()
test_data = codon_loader.get_test_data()

# Example printing the sizes of the training and test data:
print(f"Training data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")
