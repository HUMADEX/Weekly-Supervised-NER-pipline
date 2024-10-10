import pandas as pd
import glob
import ast


# Function to convert the string to a list of integers
def convert_to_int_list(tag_string):
    # Remove the square brackets and split by whitespace
    tag_list = tag_string.strip('[]').split()
    # Convert each element in the list to an integer
    return [int(i) for i in tag_list]

def convert_to_word_list(sentence_string):
    word_list = ast.literal_eval(sentence_string)
    return word_list

language = "language_code"
pattern = "pattern"

path = "/".join([language, pattern])

files = glob.glob(path)

dataframes = [pd.read_csv(f) for f in files]

combined_df = pd.concat(dataframes, ignore_index=True)

combined_df = combined_df[combined_df['sentence'] != '[]']

# Apply the function to the 'tags' column
combined_df['tags'] = combined_df['tags'].apply(convert_to_int_list)

# Apply the function to the 'sentence' column
combined_df['sentence'] = combined_df['sentence'].apply(convert_to_word_list)

combined_df.to_parquet("path.parquet", index=False)