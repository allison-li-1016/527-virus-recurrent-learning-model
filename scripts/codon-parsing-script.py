from Bio import SeqIO
import os

with open(os.path.join("../data/","resulting-codons.txt"),"w") as f:
        directory = os.fsencode("../data/parsed_gene_sequences/")
        for file in os.scandir(directory):
            if os.path.isfile(file):
                for seq_record in SeqIO.parse(file, "fasta"):
                        f.write(str(seq_record.id) + "\n")
                        for e in range(0, len(seq_record.seq),3):
                            f.write(str(seq_record.seq[e: e + 3]) + ", ")  
                        f.write(str("\n"))