import pandas as pd
from tqdm import tqdm
import numpy as np
import multiprocessing
import torch
from process_notes import process_and_update  # Import the function

def process_chunk(args):
    i, chunk = args
    result = process_and_update((i, chunk))
    return i, result

def log_gpu_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2} MB")

if __name__ == '__main__':
    try:
        torch.cuda.empty_cache()
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Load the DataFrame
    df = pd.read_parquet("path_to_file")
    # Filter out rows where length of 'text' column is greater than 5000
    df = df[df['text'].apply(len) <= 5000]

    # Optionally, you can reset the index of the filtered DataFrame if needed
    df.reset_index(drop=True, inplace=True)
    print("Data loaded from parquet.")

    # Split DataFrame into chunks
    total_chunks = 260  # More chunks
    chunks = np.array_split(df['text'], total_chunks)
    # chunks = chunks[90:]
    # Number of parallel processes
    num_processes = 10  # Less than the number of chunks

    # Create a progress bar for the main process
    main_progress_bar = tqdm(total=len(df), desc="Processing notes")

    def save_chunk_result(i, result):
        iob_tags_t, iob_tags_b, encoded_tags = zip(*result)
        chunk_df = pd.DataFrame({
            'text': chunks[i],
            'IOB_t': iob_tags_t,
            'IOB_b': iob_tags_b,
            'encoded_tags': encoded_tags
        })
        chunk_df.to_parquet(f"data/final_corpus_tagged1_chunk_{i}.parquet")
        main_progress_bar.update(len(result))
        torch.cuda.empty_cache()  # Free up memory after processing each chunk
        log_gpu_memory()  # Log memory usage after saving the chunk

    # Use multiprocessing to parallelize the tagging process
    with multiprocessing.get_context('spawn').Pool(processes=num_processes) as pool:
        for i, result in pool.imap_unordered(process_chunk, [(i, chunks[i]) for i in range(90, total_chunks)]):
            log_gpu_memory()
            save_chunk_result(i, result)

    print("Processing complete. DataFrame saved in chunks.")
