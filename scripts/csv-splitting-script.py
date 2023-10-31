import pandas as pd
import os

#csv file name to be read in
in_csv = '../data/all_sequences_metadata.csv'
new_dir = '../data/split_sequences_metadata'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
df = df = pd.read_csv(in_csv)
header = df.columns
#get the number of lines of the csv file to be read
number_lines = sum(1 for row in (open(in_csv)))

#size of rows of data to write to the csv,

#you can change the row size according to your need
rowsize = 10000

#start looping through data writing it to a new file for each set
for i in range(0,number_lines,rowsize):

    df = pd.read_csv(in_csv,
          nrows = rowsize,#number of rows to read at each loop
          skiprows = i)#skip rows that have been read
    df.columns = header
    #csv to write data to a new file with indexed name. input_1.csv etc.
    out_csv = 'split_sequences_' + str(i) + '.csv'

    df.to_csv(os.path.join(new_dir,out_csv),
          index=False,
          header=True,
          mode='a',#append data to csv file
          chunksize=rowsize)#size of data to append for each loop