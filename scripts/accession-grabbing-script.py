import pandas as pd
import csv
import os

folder_path = "../data/split_sequences_metadata"
directory = os.fsencode(folder_path)
rows_for_csv = []
for f in os.listdir(directory):
    filename = os.fsdecode(f)
    filename = folder_path + "/" + filename
    df = pd.read_csv(filename)
    accessions_list = df['GenBank Accessions'].tolist()
    for i in accessions_list:
        rows_for_csv.append(i)
with open(os.path.join("../data/",'accessions.txt'), 'w') as f:
    f.write("\n".join(map(str, rows_for_csv)))

    