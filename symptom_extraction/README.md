# Dataset Building


## Data Integration and Preprocessing

We begin by merging two distinct datasets of English medical texts. This step ensures a robust and diverse corpus, combining the strengths of both datasets. Following the integration, we preprocess the texts to clean the data, which includes removal of strings that do not contain relevant information. This preprocessing step is crucial to ensure the texts are in an optimal format for subsequent annotation.

- **Dataset 1**: [Kabatubare/autotrain-data-1w6s-u4vt-i7yo](https://huggingface.co/datasets/Kabatubare/autotrain-data-1w6s-u4vt-i7yo)
- **Dataset 2**: [s200862/medical_qa_meds](https://huggingface.co/datasets/s200862/medical_qa_meds)

The data underwent a preprocessing process using the `preprocessing/preprocess.py` script.
1. **Data Cleaning**: Since our dataset consisted of question-answer pairs between a user and an assistant, some extraneous text could be removed without losing relevant information.
   - In the **Kabatubare/autotrain-data-1w6s-u4vt-i7yo** dataset, we removed the following strings:
     - `Human:`
     - `Assistant:`
     - `\n` (newline characters)
     - `\t` (tab characters)
     - Hyphens between words (`-`) were replaced with a single space.
   - In the **s200862/medical_qa_meds** dataset, we removed:
     - `[INST]`
     - `[/INST]`
     - `<s>`
     - `</s>`
     - `\n` (newline characters)
     - `\t` (tab characters)

2. **Punctuation Removal**: All punctuation marks were removed from the text to ensure consistency.

3. **Lowercasing**: Finally, the entire dataset was converted to lowercase to standardize the text.


## Annotation with Stanza's i2b2 Clinical Model

The preprocessed English texts are then annotated using [Stanza's i2b2 Clinical Model](https://stanfordnlp.github.io/stanza/available_biomed_models.html). This model is specifically designed for clinical text processing, and it annotates each text with three labels:
   - **PROBLEM**: Includes diseases, symptoms, and medical conditions.
   - **TEST**: Represents diagnostic procedures and laboratory tests.
   - **TREATMENT**: Covers medications, therapies, and other medical interventions.

This annotation step is essential for creating a labeled dataset that serves as the foundation for training and evaluating Named Entity Recognition (NER) models.

We used Stanza's clinical-domain NER system, which contains a general-purpose NER model trained on the **2010 i2b2/VA dataset**. This model efficiently extracts entities related to problems, tests, and treatments from various types of clinical notes.

The code can be found in the following files:
- `tagging/tagDocument.py`
- `tagging/process_notes.py`

Visit [weakly-supervised-multi-lingual-ner-pipeline](https://huggingface.co/collections/rigonsallauka/weakly-supervised-multi-lingual-ner-pipeline-67079f566a22b1b67ac9631f) collection in HuggingFace Hub to see the datasets.

