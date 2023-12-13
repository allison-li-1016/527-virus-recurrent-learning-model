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


### Getting started with own input files
To run the network on your own virus dataset, you will need to provide all the metadata (GenBank Accession Numbers) to your sequences in the form of a csv file as the input to our data parsing workflow.
Download the virus metadata off of the [BV-BRC Database](https://www.bv-brc.org/). Rename the resulting CSV file as 'all-sequences-metadata.csv' and drop into the 'data' folder. Drop your reference sequence fasta and GenBank file into the 'data' folder and name it as 'reference-sequence.fasta' and 'reference-sequence.gb' respectively.

Run workflow for Mac Users

`python csv-splitting-script.py; python accession-grabbing-script.py; python full-sequence-fasta-retrieval-script; ./alignment_script.sh; ./gene-parse-script.sh; python codon-parsing-script.py`

Run workflow for Windows Users

`python csv-splitting-script.py; python accession-grabbing-script.py;python full-sequence-fasta-retrieval-script; dos2unix alignment_script.sh; ./alignment_script.sh; dos2unix gene-parse-script.sh; ./gene-parse-script.sh; python codon-parsing-script.py`

Alternatively, if you already have the list of accessions, you can forgo some of the above steps for an expedited pipeline. 

Run expedited workflow for Mac Users

`./alignment_script.sh; ./gene-parse-script.sh; python codon-parsing-script.py`

Run expedited workflow for Windows Users

`dos2unix alignment_script.sh; ./alignment_script.sh; dos2unix gene-parse-script.sh; ./gene-parse-script.sh; python codon-parsing-script.py`

<p align="center">
     <img src="images/data-processing-workflow.png" alt="workflow diagram for data processing steps" width="600"/>
</p>

## Data Curation
All sequence data is from Genbank. We filtered for human coronavirus genomes that are at least 80% sequenced in the given gene of interest.

## Model Architecture and Pre-Training
### Model Design 
We built a very simple recurrent neural network architecture that involves a RNN layer, a ReLU activation function, a fully connected layer to map to the number of features that we have, and a softmax layer to get the probabilities. The type of RNN layer, hidden layer size, and number of layers for the RNN layer can be tuned as hyperparameters.


<p align="center">
     <img src="images/model-architecture.png" alt="diagram of the model layers" width="600"/>
</p>

### Pre-training Techniques
We have two different types of model to represent each pre-training technique that we used to assess pre-training effects on model performance. 
The masked language modeling approach randomly masks a given percentage of the input sequence (15 percent by default). The model takes the masked input sequence, and predicts for the original unmasked sequence. 

<p align="center">
     <img src="images/masked.png" alt="diagram of the model layers" width="600"/>
</p>

In the autoregressive model, for every codon in the sequence, the model tries to predict for the next codon in the sequence. The input of the model would be the original sequence, and the predicted output would be the sequence shifted by one. This is similar to N-grams. 

<p align="center">
     <img src="images/autoregressive.png" alt="diagram of the model layers" width="600"/>
</p>

### Running the model

To run a hyperparameter grid search of the models:

`python main.py`

## Paper
Experiments ran with this model and subsequent results can be viewed within this [final report linked here](https://drive.google.com/file/d/1n7V0kR3fg1HvK_lpSMG0ikOYdubCQBsa/view?usp=sharing). 