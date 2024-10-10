import stanza
from tqdm import tqdm


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

def process_notes_chunk(notes_chunk, chunk_index):
    print(f"Process {chunk_index} started.")
    chunk_results = []
    nlp = stanza.Pipeline('en', package='i2b2', processors={'tokenize': 'ewt','ner': 'i2b2'}, logging_level='ERROR', device='cuda:0', use_gpu=True)  # Initialize stanza within each process
    
    # Create a progress bar for the chunk
    chunk_progress_bar = tqdm(total=len(notes_chunk), desc=f"Chunk {chunk_index}")

    for note in notes_chunk:
        doc = nlp(note)
        note_tagged_t = ""
        note_tagged_b = ""
        encoded_tags = []
        for sentence in doc.sentences:
            for token in sentence.tokens:
                note_tagged_t += f"{token.text}\t{token.ner}\n"
                note_tagged_b += f"{token.text} ({token.ner}) "
                encoded_tags.append(tag_encodings[token.ner])
        chunk_results.append((note_tagged_t, note_tagged_b, encoded_tags))
        chunk_progress_bar.update(1)  # Update the chunk progress bar

    chunk_progress_bar.close()
    print(f"Process {chunk_index} completed.")
    return chunk_results

def process_and_update(args):
    chunk_index, chunk = args
    print(f"Starting processing of chunk {chunk_index}")
    result = process_notes_chunk(chunk, chunk_index)
    print(f"Completed processing of chunk {chunk_index}")
    return result