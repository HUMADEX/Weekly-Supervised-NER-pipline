## Developing a Robust Multilingual NER Model

To develop a robust NER model capable of extracting medical entities across different languages, we implemented the training process explained below. This process encompasses data augmentation, dataset splitting, model configuration, training, and evaluation.

### 1. Data Augmentation
The data augmentation process is implemented in the `data_processing/data_augmentation.py` script. Before splitting the dataset, we performed data augmentation to enhance the diversity and robustness of the training data. This process involved two main strategies:
   - **Sentence Reordering**: Words within each sentence were reordered to create new variations of the same sentence structure. This method increases the variability of the dataset, enabling the model to generalize better to different sentence formations.
   - **Entity Extraction**: All words within each sentence that were annotated with non-"O" labels (i.e., labeled as `PROBLEM`, `TEST`, or `TREATMENT`) were extracted and used to generate new sentences. These sentences were then added back into the dataset, ensuring that the model would encounter more examples of key medical entities during training.

### 2. Dataset Splitting
The dataset splitting process is handled by the `data_processing/processing_and_splitting.py` script. After augmentation, the dataset was split into three distinct sets:
   - **Training Set (80%)**: The largest portion of the dataset was allocated for training the model. This set was used to fine-tune the model's weights based on the annotated examples.
   - **Validation Set (10%)**: A smaller portion was reserved for validation during training. This set enabled the model to be evaluated periodically, allowing us to monitor its performance and apply early stopping if necessary.
   - **Test Set (10%)**: The final portion was held out as a test set. This set was used for evaluating the model's performance after training, providing an unbiased measure of its ability to generalize to unseen data.

### 3. Model Configuration
We configured a BERT-based NER model with specific parameters to optimize its performance:
   - **Architecture**: The BERT-base-cased model was selected for its strong performance on NER tasks, particularly in handling multilingual text.
   - **Labels**: Custom labels were defined for the entities of interest, including various tags for `PROBLEM`, `TEST`, and `TREATMENT` entities.
   - **Training Parameters**: The model was set to train for 200 epochs with a batch size of 64. An AdamW optimizer was used, with a learning rate of 3e-5 and a weight decay of 0.01 to prevent overfitting.
   - **Loss Function**: A focal loss function was employed to address class imbalance by focusing more on harder-to-classify examples.

### 4. Training Process
The training process is executed from the `train/` folder using the following command:

```bash
singularity exec -B /raid/umutarioz/:/Dockers/ --nv /raid/umutarioz/BERT/ner_env.sif/ bash /Dockers/BERT/create_ner_env.sh data/train.parquet data/validate.parquet data/test.parquet "0,1" 2 "my_run_name" "log_file_name"
```

Visit [weakly-supervised-multi-lingual-ner-pipeline](https://huggingface.co/collections/rigonsallauka/weakly-supervised-multi-lingual-ner-pipeline-67079f566a22b1b67ac9631f) collection in HuggingFace Hub to see the models.