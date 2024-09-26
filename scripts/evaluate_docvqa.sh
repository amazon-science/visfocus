#!/bin/bash


export PYTHONPATH=$(pwd)

if [ "$1" == "TEST" ]; then
    IS_TEST_MODE=True
    echo "Test Mode"
else
    IS_TEST_MODE=False
fi


# Get the experiment directory path from the command-line argument
CKPT_PATH=checkpoints/vf_base_docvqa_v1
experiment_directory=checkpoints/vf_base_docvqa_v1

: > $CKPT_PATH/eval_logs.log

# run python
command="torchrun \
--nproc_per_node=8 \
scripts/eval_vqa_model_from_config.py \
--output_dir=$CKPT_PATH \
--model_config_path=$experiment_directory/vqa_model_args.yaml \
--data_config_path=$experiment_directory/vqa_data_args.yaml \
--do_train=False \
--do_eval=True \
--do_predict=$IS_TEST_MODE \
--dataloader_num_workers=8 \
--per_device_eval_batch_size=1 \
--bf16=False \
--fp16=False \
--ignore_data_skip \
--resume_from_checkpoint=$CKPT_PATH/checkpoint-15000 \
--evaluation_strategy=steps"

# echo -e $command
echo "Logfile: $CKPT_PATH/eval_logs.log"
eval nohup $command > $CKPT_PATH/eval_logs.log 2>&1

# if [ "$IS_TEST_MODE" == "True" ]; then
#   mkdir -p $CKPT_PATH/test_logs
#   eval nohup python scripts/create_json_for_submission.py --dir_path=$CKPT_PATH/test_logs > "$CKPT_PATH/eval_logs_test.log" 2>&1
# fi
