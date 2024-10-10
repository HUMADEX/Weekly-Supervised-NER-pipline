#!/bin/bash
# run code within sif within conda env: ner_env #
# within singularity shell do:
eval "$(conda shell.bash hook)"
conda env list
# conda activate ner_env
conda activate ner_env_2
cd /Dockers/BERT/
bash NER_training.sh "$@" & 