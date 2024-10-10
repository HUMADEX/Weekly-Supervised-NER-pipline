"""
This file does the data augmentation in three steps:
    1. Removes all rows where tags are vectors consisting of all 0s,
    2. Rearranges the words within each sentence, along with the respective tags,
        and saves them as new entries in the dataset,
    3. Extracts only non-0 tagged words from each sentence and saves them as new
        entries in the dataset,
The new dataset is than saved in a new path.
"""

import pandas as pd
import random

# Function to rearrange sentences
def rearrange_sentence(sentence, tags):
    combined = list(zip(sentence, tags))
    random.shuffle(combined)
    new_sentence, new_tags = zip(*combined)
    return list(new_sentence), list(new_tags)


all_rows = pd.read_parquet("path_to_file")

# Data augmentation
augmented_sentences = []
extracted_words = []

for i, row in all_rows.iterrows():
    sentence = row['sentence']
    tags = row['tags']
    
    if any(tag != 0 for tag in tags):
        # Rearrange the sentence
        new_sentence, new_tags = rearrange_sentence(sentence, tags)
        augmented_sentences.append({"sentence": new_sentence, "tags": new_tags})
        
        # Extract all words with non-zero tags
        non_zero_words = [word for word, tag in zip(sentence, tags) if tag != 0]
        non_zero_tags = [tag for tag in tags if tag != 0]
        
        if non_zero_words:
            extracted_words.append({"sentence": non_zero_words, "tags": non_zero_tags})

# Step 1: Identify rows where 'tags' is a list of all zeros
rows_to_remove = all_rows['tags'].apply(lambda x: all(tag == 0 for tag in x))

# Step 2: Remove those rows
all_rows = all_rows[~rows_to_remove]

# Convert lists to DataFrame
augmented_df = pd.DataFrame(augmented_sentences)
extracted_words_df = pd.DataFrame(extracted_words)

# Reset the index (optional, but often useful)
all_rows = all_rows.reset_index(drop=True)

# Concatenate the original, augmented, and extracted DataFrames
final_df = pd.concat([all_rows, augmented_df, extracted_words_df], ignore_index=True)

shuffled_df = final_df.sample(frac=1).reset_index(drop=True)

shuffled_df.to_parquet("path_to_save")
