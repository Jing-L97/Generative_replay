# segment data to prepare for training
import os
import glob
import pandas as pd
from tqdm import tqdm

# Set directory and output sizes
input_dir = '/scratch2/jliu/Generative_replay/knn-transformers/data/train/raw'  # Replace with your directory path
#output_sizes = [10000000, 20000000,30000000,40000000,50000000,55000000]  # Sizes of concatenated output files
output_sizes = {'10M':10000000, '20M':20000000, '30M':30000000, '40M':40000000,'50M':50000000,'55M':55000000}

output_files = []  # To store paths of output files

# Step 1: Gather file information
file_word_counts = {}
total_words = 0

# Read all text files in the directory
for filepath in glob.glob(os.path.join(input_dir, '*.train')):
    with open(filepath, 'r') as f:
        data = f.readlines()
        word_count = sum(len(line.split()) for line in data)
        file_word_counts[filepath] = word_count
        total_words += word_count

# Step 2: Calculate proportions and sizes for each output file
file_proportions = {filepath: word_count / total_words for filepath, word_count in file_word_counts.items()}

# Step 3: Create concatenated output files based on specified sizes
for size_name, output_size in output_sizes.items():
    output_file_path = os.path.join(input_dir, f'{size_name}.txt')
    output_files.append(output_file_path)

    words_to_add = output_size
    current_size = 0
    output_data = []

    # Iterate through each file to gather data based on its proportion
    for filepath, proportion in file_proportions.items():
        if current_size >= words_to_add:
            break

        # Calculate the number of words to take from this file
        words_from_file = int(output_size * proportion)
        remaining_space = words_to_add - current_size

        # If the calculated words to take exceed the remaining space, take only what fits
        if words_from_file > remaining_space:
            words_from_file = remaining_space

        # Read the file and add the specified number of words
        with open(filepath, 'r') as f:
            lines = f.readlines()
            words_added = 0
            for line in lines:
                line_words = line.split()
                line_word_count = len(line_words)

                if words_added + line_word_count > words_from_file:
                    # Take only part of this line to reach the limit
                    output_data.append(' '.join(line_words[:words_from_file - words_added]))
                    break  # Stop after reaching the desired word count

                output_data.append(line)
                words_added += line_word_count

            current_size += words_added

    # Write the concatenated data to the output file
    with open(output_file_path, 'w') as out_file:
        out_file.write('\n'.join(output_data))

# Output results
for size, file_path in zip(output_sizes, output_files):
    print(f'Created: {file_path} with size: {size} words')
