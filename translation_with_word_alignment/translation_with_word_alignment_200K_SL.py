import pandas as pd
import numpy as np
import torch
import transformers
import itertools
import copy

import json
#import requests
import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartForConditionalGeneration, MBart50TokenizerFast

if torch.cuda.is_available():
    print("Running on GPU")
else:
    print("Running on CPU")

# LOADING TRANSLATION MODEL

tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


# LOADING WORD ALIGNMENT ALGORITHM FOR THE LABELS

word_alignment_model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
word_alignment_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# TRANSLATION MODEL

def translation_model (sample_sentence):
    
    #print('sample_sentence: ', sample_sentence)
    tokenizer.src_lang = "en_XX"
    
    encoded_hi = tokenizer(sample_sentence, return_tensors="pt")
    
    generated_tokens = model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer.lang_code_to_id["sl_SI"]
    )
    
    translation_sl = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    #print('translation_sl: ', translation_sl)
    
    # Ensure translation_sl is a string
    if isinstance(translation_sl, list):
        translation_sl = ' '.join(translation_sl)
    
    #print('translation_sl: ', translation_sl)
    
    return translation_sl


# WORD ALIGNMENT

def word_alignment (src,tgt,sample_labels):
    #print('5 before splitting wa : ',tgt)
    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    #print('6 after splitting wa : ',sent_tgt)

    token_src, token_tgt = [word_alignment_tokenizer.tokenize(word) for word in sent_src], [word_alignment_tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [word_alignment_tokenizer.convert_tokens_to_ids(x) for x in token_src], [word_alignment_tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = word_alignment_tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=word_alignment_tokenizer.model_max_length, truncation=True)['input_ids'], word_alignment_tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=word_alignment_tokenizer.model_max_length)['input_ids']

    sub2word_map_src = []

    for i, word_list in enumerate(token_src):
      sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []

    for i, word_list in enumerate(token_tgt):
      sub2word_map_tgt += [i for x in word_list]

    # alignment

    align_layer = 8
    threshold = 1e-3

    word_alignment_model.eval()

    with torch.no_grad():
    
      out_src = word_alignment_model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
      out_tgt = word_alignment_model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

      dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

      softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
      softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

      softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()

    for i, j in align_subwords:
        if i <= 180:
            align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )


    new_target_labels = np.zeros((len(sent_tgt)), dtype=int)    
    
    #print(align_words)

    for i, j in sorted(align_words):
        #print(sample_labels[i])
        #print(i)
    
        if int(sample_labels[i]) == 1:
            #print('one')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 1
            sent_tgt[j] = sent_tgt[j] + '*'
        
        if int(sample_labels[i]) == 2:
            #print('two')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 2
            sent_tgt[j] = sent_tgt[j] + '*'
       
        if int(sample_labels[i]) == 3:
            #print('three')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 3
            sent_tgt[j] = sent_tgt[j] + '*'
        
        if int(sample_labels[i]) == 4:
            #print('four')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 4
            sent_tgt[j] = sent_tgt[j] + '*'
        
        if int(sample_labels[i]) == 5:
            #print('five')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 5
            sent_tgt[j] = sent_tgt[j] + '*'
        
        if int(sample_labels[i]) == 6:
            #print('six')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 6
            sent_tgt[j] = sent_tgt[j] + '*'
       
        if int(sample_labels[i]) == 7:
            #print('seven')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 7
            sent_tgt[j] = sent_tgt[j] + '*'
        
        if int(sample_labels[i]) == 8:
            #print('eight')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 8
            sent_tgt[j] = sent_tgt[j] + '*'
        
        if int(sample_labels[i]) == 9:
            #print('nine')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 9
            sent_tgt[j] = sent_tgt[j] + '*'
        
        if int(sample_labels[i]) == 10:
            #print('ten')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 10
            sent_tgt[j] = sent_tgt[j] + '*'
       
        if int(sample_labels[i]) == 11:
            #print('eleven')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 11
            sent_tgt[j] = sent_tgt[j] + '*'
        
        if int(sample_labels[i]) == 12:
            #print('twelve')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 12
            sent_tgt[j] = sent_tgt[j] + '*'
            
        if int(sample_labels[i]) not in (1,2,3,4,5,6,7,8,9,10,11,12):
            #print('twelve')
            index = sent_tgt.index(sent_tgt[j])
            new_target_labels[index] = 0
            sent_tgt[j] = sent_tgt[j] + '*'
    
    return new_target_labels


# ALL TRANSLATION PROCESS

def translation(sample, sample_labels):
    #max_length = 4999
    max_sequence_length = 512  # Maximum sequence length for the model

    def chunk_text(text, max_len):
        chunks = []
        while len(text) > max_len:
            chunks.append(text[:max_len])
            text = text[max_len:]
        chunks.append(text)
        return chunks

    translated_chunks = []
    for chunk in chunk_text(" ".join(sample), max_sequence_length):
        translated_chunks.append(translation_model(chunk))

    # Combine translated chunks
    #print('translated_chunks: ',translated_chunks)
    tgt = " ".join(translated_chunks)

    # ... rest of your code for word alignment and other processing

    src = " ".join(sample)  # original sentence

    new_target_labels = word_alignment(src, tgt, sample_labels)  # Fixed indentation (use spaces)
    sent_target = tgt.strip().split()

    return sent_target, new_target_labels

# Load the dataset
df_original_dataset = pd.read_parquet("main_dataset_english_stanza.parquet")
df_original_dataset = df_original_dataset[200000:len(df_original_dataset)]


# chunk process

# Define the function to process the chunks
def process_chunks(df, chunk_size=10000):
    num_rows = len(df)
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        df_chunk = df.iloc[start:end]
        df_translated_chunk = pd.DataFrame(columns=["sentence", "tags"])

        for index, row in df_chunk.iterrows():
            #print(f'Processing row {index}')
            sample = row['sentence']
            
            if len(sample) > 5000:
                continue # skip to the next iteration
            
            sample_labels = row['tags']

            # Perform translation (you need to define this function)
            sent_tgt, new_target_labels = translation(sample, sample_labels)

            # Store the translation in the new DataFrame
            df_translated_chunk.at[index, 'sentence'] = sent_tgt
            df_translated_chunk.at[index, 'tags'] = new_target_labels

        # Save the translated chunk to a CSV file
        chunk_filename = f'subset_2OOK_main_dataset_translated_SL_10K_chunk_after_200K_{start // chunk_size}.csv'
        df_translated_chunk.to_csv(chunk_filename, index=False)
        #print(f'Saved: {chunk_filename}')

# Call the chunk processing function
process_chunks(df_original_dataset)
