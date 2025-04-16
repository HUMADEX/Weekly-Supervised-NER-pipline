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

Polish: 
Model name: gsarti/opus-mt-tc-en-pl(https://huggingface.co/gsarti/opus-mt-tc-en-pl) 

Spanish:
Model name: Helsinki-NLP/opus-mt-tc-big-en-es (https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-es)

Greek: 
Model name: Helsinki-NLP/opus-mt-en-el (https://huggingface.co/Helsinki-NLP/opus-mt-en-el)

Portugese: 
Model name: Helsinki-NLP/opus-mt-tc-big-en-pt (https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt)

Italian: 
Model name: Helsinki-NLP/opus-mt-tc-big-en-it (https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-it)

German: 
Model name: Helsinki-NLP/opus-mt-en-de (https://huggingface.co/Helsinki-NLP/opus-mt-en-de) 

Slovenian: 
Model name: facebook/mbart-large-50-many-to-many-mmt (https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) 

### Word Alignment

This approach leverages contextual word embeddings and different techniques to find corresponding words between sentences in different languages.

Model: [aneuraz/awesome-align-with-co](https://huggingface.co/aneuraz/awesome-align-with-co)

Reference: Dou, Z. Y., & Neubig, G. (2021). Word alignment by fine-tuning embeddings on parallel corpora. arXiv preprint arXiv:2101.08231.

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

Visit [weakly-supervised-multi-lingual-ner-pipeline](https://huggingface.co/collections/HUMADEX/weakly-supervised-multi-lingual-ner-pipeline-67091a099e653e1af93a352a) collection in HuggingFace Hub to see the models and datasets.

## Preprint
The preprint version of the paper associated with this code is available at:
[Preprint DOI: 10.20944/preprints202504.1356.v1](https://www.preprints.org/manuscript/202504.1356/v1)

## Acknowledgement

This code had been created as part of joint research of HUMADEX research group (https://www.linkedin.com/company/101563689/) and has received funding by the European Union Horizon Europe Research and Innovation Program project SMILE (grant number 101080923) and Marie Skłodowska-Curie Actions (MSCA) Doctoral Networks, project BosomShield (grant number 101073222). Responsibility for the information and views expressed herein lies entirely with the authors.

Authors:
dr. Izidor Mlakar, Rigon Sallauka, dr. Umut Arioz, dr. Matej Rojc
