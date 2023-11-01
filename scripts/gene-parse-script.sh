#!/bin/bash

mkdir -p ../data/parsed_gene_sequences/
rm -rf ../data/aligned_sequences/*/
for file in ../data/aligned_sequences/*
do
  filename=`basename $file .fasta`  # Extract the filename without path
  output_file="../data/parsed_gene_sequences/${filename}_parsed.fasta"
  python gene-parsing-script.py --alignment "$file" --reference "../data/reference_sequence.gb" --output "$output_file" --percentage 0.8 --gene 'S'
done