import requests
import os
with open("../data/accessions.txt") as acc_lines:
        accs = [line.strip() for line in acc_lines]
op_dir = os.path.abspath("../data/full_sequence_fastas/")
if not os.path.exists(op_dir):
    os.makedirs(op_dir)
for i in accs:
    print(i)
    url = "http://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?db=nuccore&dopt=fasta&sendto=on&id=" + i
    r = requests.get(url)
    print("request received for " + i)
    fname = "full_sequence_" + i + ".fasta"
    open(os.path.join(op_dir,fname) , 'wb').write(r.content)
    print("wrote record for  " + i)