#!/bin/bash


export PYTHONPATH=$(pwd)

# Check if an experiment directory path is provided as an argument
if [ $# -ge 3 ]; then
  echo "Usage: $0 <experiment_directory_path> <tail>"
  exit 1
fi

# Get the experiment directory path from the command-line argument
experiment_directory="$1"

# Check if the experiment directory exists
if [ ! -d "$experiment_directory" ]; then
  echo "Error: Experiment directory does not exist."
  exit 1
fi

# List all directories within the experiment directory starting with 'checkpoint-'
# checkpoint_directories=("$experiment_directory"/checkpoint-*)

: > $experiment_directory/nohup_eval_all_checkpoints.log
echo -e "\nLogging (stdout+stderr) file:\n\t$experiment_directory/nohup_eval_all_checkpoints.log"

echo -e "\nEvaluating the follwing ckpts:"
for checkpoint_dir in $(find $experiment_directory -type d -name "checkpoint-*" | sort -V | tail -$2); do
  echo -e "\t$checkpoint_dir"
done

# Iterate over the checkpoint directories
# for checkpoint_dir in "${checkpoint_directories[@]}"; do
for checkpoint_dir in $(find $experiment_directory -type d -name "checkpoint-*" | sort -V | tail -$2); do
  if [ -d "$checkpoint_dir" ]; then
    echo -e "\nEvaluation: $checkpoint_dir"

    # run python
    command="torchrun \
    --nproc_per_node=8 \
    scripts/eval_vqa_model_from_config.py \
    --output_dir=$experiment_directory \
    --model_config_path=$experiment_directory/vqa_model_args.yaml \
    --data_config_path=./configs/configs_ofir/infovqa_wds_data.yaml \
    --do_train=False \
    --do_eval=True \
    --dataloader_num_workers=8 \
    --per_device_eval_batch_size=1 \
    --bf16=False \
    --fp16=True \
    --ignore_data_skip \
    --resume_from_checkpoint=$checkpoint_dir \
    --evaluation_strategy=steps"
    # --run_name=oregon_text-haifa-ofirab-p3dn-4/expts/models/swinv2_t5endec/swin_small_t5_base/finetune_docvqa_idl_v2_mpm_L0123Al05_idl_v2_accum4_f32__noWD
    # --per_device_train_batch_size=2
    # --learning_rate=5e-5
    # --lr_scheduler_type=cosine
    # --weight_decay=0.00
    # --max_steps=200000
    # --save_steps=10000
    # --save_total_limit=20
    # --disable_tqdm=true
    # --num_train_epochs=50
    # --warmup_steps=1000
    # --eval_steps=100000
    # --gradient_accumulation_steps=1
    # --ddp_find_unused_parameters=False
    
    # echo $command
    eval nohup $command >> $experiment_directory/nohup_eval_all_checkpoints.log 2>&1

  fi
done
