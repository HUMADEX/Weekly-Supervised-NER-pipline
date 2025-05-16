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

## Publication
The paper associated with this code has been published: [10.3390/app15105585](https://doi.org/10.3390/app15105585)

Please cite this paper as follows if you use this code or build upon this work. Your citation supports the authors and the continued development of this research.
```bibtex
@article{app15105585,
  author  = {Sallauka, Rigon and Arioz, Umut and Rojc, Matej and Mlakar, Izidor},
  title   = {Weakly-Supervised Multilingual Medical NER for Symptom Extraction for Low-Resource Languages},
  journal = {Applied Sciences},
  volume  = {15},
  year    = {2025},
  number  = {10},
  article-number = {5585},
  url     = {https://www.mdpi.com/2076-3417/15/10/5585},
  issn    = {2076-3417},
  doi     = {10.3390/app15105585}
}
```


## Acknowledgement

This code had been created as part of joint research of HUMADEX research group (https://www.linkedin.com/company/101563689/) and has received funding by the European Union Horizon Europe Research and Innovation Program project SMILE (grant number 101080923) and Marie Skłodowska-Curie Actions (MSCA) Doctoral Networks, project BosomShield (grant number 101073222). Responsibility for the information and views expressed herein lies entirely with the authors.

Authors:
dr. Izidor Mlakar, Rigon Sallauka, dr. Umut Arioz, dr. Matej Rojc
