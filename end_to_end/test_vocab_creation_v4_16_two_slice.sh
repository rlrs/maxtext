#!/bin/bash
set -e

RUN_NAME=${1}_$(date +%Y-%m-%d-%H)
OUTPUT_PATH=${2}
DATASET_PATH=${3}
VOCAB_PATH=vocab_test_creation_$RUN_NAME


#Setup and Train
bash setup.sh

python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME steps=5 dcn_data_parallelism=2 ici_fsdp_parallelism=8\
    enable_checkpointing=False base_output_directory=$OUTPUT_PATH dataset_path=$DATASET_PATH vocab_relative_path=$VOCAB_PATH

python3 end_to_end/eval_assert.py 0 0 $OUTPUT_PATH/$VOCAB_PATH vocab_creation
