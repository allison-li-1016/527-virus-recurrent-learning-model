#!/bin/bash

mkdir -p ../data/aligned_sequences/
for file in ../data/full_sequence_fastas/*
do
  filename=`basename $file .fasta`  # Extract the filename without path
  output_file="../data/aligned_sequences/${filename}_aligned.fasta"
  augur align --sequences "$file" --reference-sequence "../data/reference_sequence.fasta" --output "$output_file" --fill-gaps --nthreads 4
  find ../data/aligned_sequences/. -type f  ! -name "*.fasta" -exec rm {} \;
done