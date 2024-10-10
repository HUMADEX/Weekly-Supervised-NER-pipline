# Weakly Supervised NER pipeline

This repository contains the Named Entity Recognition (NER) pipeline for symptom extraction, focusing on multilingual capabilities. The pipeline leverages state-of-the-art NLP models to identify entities related to medical conditions, tests, and treatments across multiple languages.

## Overview
The NER pipeline is designed to automatically annotate medical text in English and extend this functionality to seven additional languages. The core components include annotation using a clinical model, translation of the annotated text, and fine-tuning language-specific models.

## Pipeline Workflow
The symptom extraction NER pipeline consists of the following key steps:

### Annotation of English Data
English medical texts are annotated using the Stanza clinical model.
The following entity tags are used: PROBLEM, TEST, and TREATMENT.


### Translation into Multiple Languages
The annotated English dataset is translated into seven languages:
German
Italian
Spanish
Greek
Slovenian
Polish
Portuguese

### Fine-tuning Multilingual BERT Models
Language-specific BERT models are fine-tuned using the translated datasets.
The fine-tuning process adapts each model to recognize symptoms, tests, and treatments in its respective language.

### Supported Languages
The pipeline supports the following languages:

- **English (base language)**
- **German**
- **Italian**
- **Spanish**
- **Greek**
- **Slovenian**
- **Polish**
- **Portuguese**

Visit [weakly-supervised-multi-lingual-ner-pipeline](https://huggingface.co/collections/rigonsallauka/weakly-supervised-multi-lingual-ner-pipeline-67079f566a22b1b67ac9631f) collection in HuggingFace Hub to see the models and datasets.