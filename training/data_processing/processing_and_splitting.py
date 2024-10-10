"""
This file does the splitting of the dataset into train/validate/test sets.
After this it prepares each set in the correct format for training, that 
is the IOB format:

    sentenceID  word    tag
        0       word1   tag1
        0       word2   tag2
        1       word1   tag1
        1       word2   tag2
        .         .       .
        .         .       .
        .         .       

"""

from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

tag_encodings = {
    "O": 0,
    "B-PROBLEM": 1,
    "I-PROBLEM": 2,
    "E-PROBLEM": 3,
    "S-PROBLEM": 4,
    "B-TREATMENT": 5,
    "I-TREATMENT": 6,
    "E-TREATMENT": 7,
    "S-TREATMENT": 8,
    "B-TEST": 9,
    "I-TEST": 10,
    "E-TEST": 11,
    "S-TEST": 12
}

# Function to find the key for a given value
def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None 

def create_df(df):
    print("Creating new DataFrame with sentences and tags...")
    # Initialize variables
    sentence_id = 0
    data = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        sentence = row["sentence"]
        tags = row["tags"]

        # Iterate over each word and its tag
        for word, tag in zip(sentence, tags):
            iob = get_key_from_value(tag_encodings, tag)
            data.append([sentence_id, word, iob])

        # Increment sentence ID
        sentence_id += 1

    # Create a DataFrame from data
    df = pd.DataFrame(data, columns=["sentence_id", "words", "labels"])
    
    return df


def main(input_path, train_save_path, validate_save_path, test_save_path):
    print(f"Loading data from {input_path}...")
    # Load the DataFrame
    df = pd.read_parquet(input_path)
    print("Data loaded successfully.")

    # Perform train-temp split
    print("Performing initial train-temp split...")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    print("Initial train-temp split completed.")

    # Perform temp split into validate and test
    print("Performing temp split into validate and test sets...")
    validate_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print("Temp split completed.")
    
    # Create train, validate, and test DataFrames
    print("Creating train DataFrame...")
    df_train = create_df(train_df)
    print("Train DataFrame created.")
    
    print("Creating validate DataFrame...")
    df_validate = create_df(validate_df)
    print("Validate DataFrame created.")

    print("Creating test DataFrame...")
    df_test = create_df(test_df)
    print("Test DataFrame created.")

    # Save the DataFrames to parquet files
    print(f"Saving train DataFrame to {train_save_path}...")
    df_train.to_parquet(train_save_path)
    print("Train DataFrame saved successfully.")
    
    print(f"Saving validate DataFrame to {validate_save_path}...")
    df_validate.to_parquet(validate_save_path)
    print("Validate DataFrame saved successfully.")
    
    print(f"Saving test DataFrame to {test_save_path}...")
    df_test.to_parquet(test_save_path)
    print("Test DataFrame saved successfully.")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process and split dataset")
    parser.add_argument("input_path", type=str, help="Path to the input Parquet file")
    parser.add_argument("train_save_path", type=str, help="Path to save the train Parquet file")
    parser.add_argument("validate_save_path", type=str, help="Path to save the validate Parquet file")
    parser.add_argument("test_save_path", type=str, help="Path to save the test Parquet file")


    # Parse arguments
    args = parser.parse_args()

    # Run the main function
    main(args.input_path, args.train_save_path, args.validate_save_path, args.test_save_path)