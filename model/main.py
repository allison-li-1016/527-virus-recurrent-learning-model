import itertools

import CodonLoader from loader
import RNNModel from model

def main():
    path = "data/resulting-codons.txt"
    codon_loader = CodonLoader(path, test_split=0.2)
    train_data = codon_loader.get_train_data() # (y)
    test_data = codon_loader.get_test_data()

    # Pretraining masking random codons
    masked_codons = codon_loader.mask(train_data, 0.1)

    # layer_types = ["LSTM", "GRU"]
    # hidden_layer_sizes = [128, 256, 512]


    # for (lt, hls) in itertools.product(layer_types, hidden_layer_sizes):
    #     model = RNNModel(layer_type, hidder_layer_size, dropout_rate, learning_rate, epochs, batch_size)
    #     # Validation data
    #     acc = model.eval()






if __name__ == "__main__":
    main()