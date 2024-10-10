# Redirect all output to a log file
# exec > >(tee -i script_output.log)
# exec 2>&1

# echo 'this directory should have bert resources ... \n'
# echo 'change content in this script, used for bert ner training ... \n'
# echo 'specify using GPU 0!! \n'

# wandb login ebddfdd3cdee610eefd05e394edfffeb25931c5e

# #!/bin/bash# Navigate to the directory containing your Python script if necessary
# cd google-bert #/path/to/python/script/directory

# Execute your Python script
# Check if the dataset paths are provided as arguments
if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <path_to_train> <path_to_validate> <path_to_test> <CUDA_VISIBLE_DEVICES> <n_gpu> <run_name> <log_file_name>"
    exit 1
fi

# Get the dataset paths from the arguments
TRAIN_PATH=$1
VALIDATE_PATH=$2
TEST_PATH=$3
CUDA_VISIBLE_DEVICES=$4
N_GPU=$5
RUN_NAME=$6
LOG_FILE_NAME=$7

# Redirect all output to the specified log file
exec > >(tee -i "$LOG_FILE_NAME")
exec 2>&1

echo 'This directory should have BERT resources ...'
echo 'Change content in this script, used for BERT NER training ...'
echo 'Specify using GPU 0!!'

# Log into Weights & Biases
wandb login ebddfdd3cdee610eefd05e394edfffeb25931c5e

# Navigate to the directory containing your Python script if necessary
cd google-bert # Replace with the actual path to your Python script directory


# Print the paths for debugging
echo "Train dataset path: $TRAIN_PATH"
echo "Validate dataset path: $VALIDATE_PATH"
echo "Test dataset path: $TEST_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs: $N_GPU"
echo "Run name: $RUN_NAME"
echo "Log file name: $LOG_FILE_NAME"

# Run the Python script with the dataset paths as arguments
echo "Running the Python script..."
python train.py "$TRAIN_PATH" "$VALIDATE_PATH" "$TEST_PATH" "$CUDA_VISIBLE_DEVICES" "$N_GPU" "$RUN_NAME" 
echo "Python script execution completed."
# Optionally, capture output or handle errors
# output=$(python3 /google-bert/simpleNER2.py 2>&1)

# Capture both standard output and error
echo "Python script output: $output"