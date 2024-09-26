#!/bin/bash


export TRAINING_STAGE="pretraining"
export PYTHONPATH=$(pwd)

EXP_ROOT_DIR=output/expts
mkdir -p $EXP_ROOT_DIR

EXP_CONF_DIR=$1

model_conf=$EXP_CONF_DIR/model_args.yaml
data_conf=$EXP_CONF_DIR/pretraining_wds_data_args.yaml
ds_conf=$EXP_CONF_DIR/ds.json
other_conf=$EXP_CONF_DIR/default_args_pretrain.txt

if [ ! -f "$model_conf" ] || [ ! -f "$data_conf" ] || [ ! -f "$ds_conf" ]; then
    echo -e "\nError: config files are not valid. Extting.\n"
    exit 1
fi

if [ ! -f "$other_conf" ]; then
    echo -e "\nWARNING: no other_args.txt specified, using default_conf.txt."
    cp scripts/default_args_pretrain.txt $EXP_CONF_DIR
fi

echo -e "\nConfirmed: all input configs are valid."

outdir="$EXP_ROOT_DIR/$(basename $EXP_CONF_DIR)"

mkdir -p $outdir
cp $EXP_CONF_DIR/* $outdir
echo -e "\nConfirmed: successfuly created experiment output directory\n\t$outdir"

command="deepspeed --num_gpus=8 \
scripts/train_vqa_model_from_config.py \
--output_dir=$outdir \
--model_config_path=$model_conf \
--data_config_path=$data_conf \
$(cat $other_conf | tr -d '\n') \
--run_name=$INSTANCE_NAME/$EXP_CONF_DIR \
--deepspeed=$ds_conf"


### Running training
echo -e "\nRunning training command:\n$(echo $command | tr ' ' '\n')" | tee $outdir/command.txt
echo -e "\nLogging (stdout+stderr) file:\n\t$outdir/nohup.log"
eval nohup $command > $outdir/nohup.log 2>&1
# echo $command
