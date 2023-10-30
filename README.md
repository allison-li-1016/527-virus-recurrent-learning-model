# Recurrent Neural Network for Understanding Sars-Cov-2 Protein Stability 

Allison Li, Jacob Evarts

University of Washington

## Introduction
Our aim is to build a recurrent neural network that takes Sars-cov-2 genomic data and reads codons of the spike protein as ‘words’ in a sentence, similar to natural language processing. By doing this across coronavirus samples and strains, the model should be able to learn not only which amino acids, but which codons typically comprise the protein. Finding patterns in the codon sequence could provide information on protein stability and codon transcription preferences by coronavirus.

### Installations
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) and mamba are required to run the workflow. After installing Miniconda, install mamba using:

`conda install mamba -n base -c conda-forge`

### Creating an Environment
Create an environment to test this Nextstrain workflow.

`mamba env create -n rnn-environment -f envs/config.yaml`

Activate the environment to use the workflow.

`conda activate rnn-environment`

Run workflow 

`snakemake --cores 4`

### Getting started with own input files
To run the network on your own virus dataset, you will need to provide all the accession numbers to your sequences in the form of a csv file as the input to our data parsing workflow.

## Data Curation
All sequence data is from Genbank. We filtered for human coronavirus genomes that are at least 80% complete. 