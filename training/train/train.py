import logging
import pandas as pd
from simpletransformers.ner import NERModel, NERArgs
import wandb
import os
import argparse
from sklearn.metrics import accuracy_score

# WANDB_PROJECT=my_project_name

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train and evaluate NER model")
parser.add_argument("train_path", type=str, help="Path to the training dataset")
parser.add_argument("validate_path", type=str, help="Path to the validation dataset")
parser.add_argument("test_path", type=str, help="Path to the test dataset")
parser.add_argument("cuda_visible_devices", type=str, help="CUDA_VISIBLE_DEVICES setting")
parser.add_argument("n_gpu", type=int, help="Number of GPUs to use")
parser.add_argument("run_name", type=str, help="Name of the run")

args = parser.parse_args()

os.environ["WANDB_PROJECT"] = "sl-tagging"
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

def compute_metrics(preds, labels):
    # Flatten predictions and labels
    preds_flat = [p for pred in preds for p in pred]
    labels_flat = [l for label in labels for l in label]

    # Calculate accuracy
    accuracy = accuracy_score(labels_flat, preds_flat)

    return accuracy

# Load datasets
train_data = pd.read_parquet(args.train_path)
eval_data = pd.read_parquet(args.validate_path)
test_data = pd.read_parquet(args.test_path)

custom_labels = [
    "O", 
    "B-PROBLEM", 
    "I-PROBLEM", 
    "E-PROBLEM", 
    "S-PROBLEM",
    "B-TREATMENT",
    "I-TREATMENT",
    "E-TREATMENT",
    "S-TREATMENT",
    "B-TEST",
    "I-TEST",
    "E-TEST",
    "S-TEST"
]

# Configure the model
model_args = NERArgs()
model_args.use_early_stopping = True
model_args.num_train_epochs = 200
model_args.train_batch_size = 64
model_args.eval_batch_size = 64
model_args.evaluate_during_training = True
model_args.early_stopping_patience = 30
model_args.output_dir = "".join(["outputs/", args.run_name])
model_args.best_model_dir = "".join([model_args.output_dir, "/best_model"])
model_args.classification_report = True
model_args.labels_list = custom_labels
model_args.overwrite_output_dir = True
model_args.wandb_project = "sl-tagging"
model_args.early_stopping_consider_epochs = True
model_args.n_gpu = args.n_gpu
model_args.optimizer = "AdamW"
model_args.reprocess_input_data = True
model_args.loss_type = "focal"
model_args.weight_decay = 0.01
model_args.learning_rate = 3e-5
model_args.save_steps = -1
model_args.save_eval_checkpoints = False
model_args.save_model_every_epoch = False

# Initialize a new wandb run
wandb.init()
learning_rate_formatted = "{:.5f}".format(model_args.learning_rate)

# Construct the run name after initializing wandb
run_name = args.run_name
wandb.run.name = run_name
wandb.run.save()

# Create a Transformer Model
model = NERModel(
    "bert", 
    "bert-base-cased", 
    args=model_args,
    use_cuda=True,
    sweep_config=wandb.config
)

# Train the model
model.train_model(train_data, eval_data=eval_data, accuracy=compute_metrics)


# Evaluate the model
model.eval_model(test_data, accuracy=compute_metrics)

    
# # Log the test results to wandb
# wandb.log(result)

# Sync wandb
wandb.join()