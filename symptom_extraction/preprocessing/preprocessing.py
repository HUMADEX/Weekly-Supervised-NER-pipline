import pandas as pd
import re

def split_into_sentences(text):
    """
    Split text into sentences based on various punctuation marks.
    
    Args:
    text (str): The input text containing multiple sentences.
    
    Returns:
    list: A list of sentences extracted from the input text.
    """
    # Split the text into sentences based on periods, question marks, exclamation marks, colons, semicolons, and ellipses
    sentences = re.split(r'[.!?;]', text)
    
    # Remove empty strings and strip whitespace from each sentence
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return sentences

# Define the function to remove text and punctuation
def remove_text(text):
    """
    Function to clean texts from kabatubare dataset

    Args:
        text (str): text from the dataset

    Returns:
        str: cleaned text
    """
    # Remove "Human:" and "Assistant:"
    text = re.sub(r'(Human|Assistant):', '', text)
    # Replace hyphen between numbers with a space
    text = text.replace('-', ' ')
    # Remove newlines and tabs
    text = text.replace('\\n', '').replace('\\t', '')
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase the text
    text = text.lower()
    return text

def clean_text(text):
    """
    Function to clean texts from medical_qa dataset

    Args:
        text (str): text from the dataset

    Returns:
        str: cleaned text
    """
    # Remove <s>, </s>, [INST], [/INST]
    text = re.sub(r'<s>|</s>|(\[INST\]|\[/INST\])', '', text)
    # Replace hyphen between numbers with a space
    text = text.replace('-', ' ')
    # Remove newlines and tabs
    text = text.replace('\n', ' ').replace('\t', ' ')
    # Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)
    # Lowercase the text
    text = text.lower()
    return text

kabatubare = pd.read_parquet("kabatubare.parquet")
medical_qa = pd.read_parquet("medical_qa.parquet")

sentence_split = kabatubare['autotrain_text'].apply(split_into_sentences)

all_sentences = []
for sentences in sentence_split:
    for sentence in sentences:
        all_sentences.append(sentence)
        
df = pd.DataFrame({'text': all_sentences})
df['text'] = df['text'].apply(remove_text)

# Optional save of the intermediary df
# df.to_parquet("kabatubare_sentences.parquet") 

medical_qa_sentences = medical_qa['text'].apply(split_into_sentences)

all_medical_qa_sentences = []
for sentences in medical_qa_sentences:
    for sentence in sentences:
        all_medical_qa_sentences.append(sentence)
        
df1 = pd.DataFrame({"text": all_medical_qa_sentences})
df1['text'] = df1['text'].apply(clean_text)

# Optional save of the intermediary df
# df1.to_parquet("medical_qa_sentences.parquet")

all_dfs = [df, df1]
concatenated_df = pd.concat(all_dfs, ignore_index=True)

# Shuffle the DataFrame
shuffled_df = concatenated_df.sample(frac=1, random_state=42).reset_index(drop=True)

shuffled_df.to_parquet("preprocessed.parquet")



