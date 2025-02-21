# import packages
import numpy as np
import cadquery as cq
from scipy.stats import qmc
import math
import itertools
import matplotlib.pyplot as plt
import os
import json
import copy
import random
import sys

# ChatGPT prompt: write a slurm script to run python script in parallel using N tasks, on different part of data stored in a text file

# Read the JSON file as a dictionary
json_filepath = '/glade/u/home/joko/spherical-code/data/s_code.json'
with open(json_filepath, 'r') as json_file:
    s_code_dict = json.load(json_file)

# Read



# Suggestion from ChatGPT below:

import sys

def process_data(input_file, start_line, end_line, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Extract the lines for the specific task based on start and end line
    chunk = lines[start_line-1:end_line]  # List slicing (1-indexed)

    # Process the chunk (this can be any processing logic you need)
    processed_chunk = [line.strip().upper() for line in chunk]  # Example: convert to uppercase

    # Write the processed chunk to the output file
    with open(output_file, 'w') as f:
        f.write("\n".join(processed_chunk))

if __name__ == "__main__":
    # Read arguments passed from the Slurm script
    input_file = sys.argv[1]
    start_line = int(sys.argv[2])
    end_line = int(sys.argv[3])
    output_file = sys.argv[4]

    # Call the function to process the data
    process_data(input_file, start_line, end_line, output_file)
