#!/bin/bash


export TRAINING_STAGE="vqa_finetune"

# check if instance name is defined. exit if not.
if [ -n "$INSTANCE_NAME" ]; then
    echo -e "\nINFO: Instance name is $INSTANCE_NAME."
else
    echo -e "\nERROR: INSTANCE_NAME is not defined. Extting."
    exit 1
fi

EXP_CONF_DIR=$1

if [ ! -d "$EXP_ROOT_DIR" ]; then
    echo -e "\nERROR: EXP_ROOT_DIR is not defined. Extting."
    exit 1
fi

echo -e "\nConfirmed: EXP_ROOT_DIR=$EXP_ROOT_DIR and is valid."

model_conf=$EXP_CONF_DIR/vqa_model_args.yaml

if [ -f "$EXP_CONF_DIR/vqa_wds_data_args.yaml" ]; then
    data_conf=$EXP_CONF_DIR/vqa_wds_data_args.yaml
else 
    data_conf=$EXP_CONF_DIR/vqa_data_args.yaml
fi

ds_conf=$EXP_CONF_DIR/ds.json
other_conf=$EXP_CONF_DIR/default_args_vqa_finetune.txt

if [ ! -f "$model_conf" ] || [ ! -f "$data_conf" ] || [ ! -f "$ds_conf" ]; then
    echo -e "\nError: config files are not valid. Extting.\n"
    exit 1
fi

if [ ! -f "$other_conf" ]; then
    echo -e "\nWARNING: no other_args.txt specified, using default_conf.txt."
    cp scripts/default_args_vqa_finetune.txt $EXP_CONF_DIR
fi

echo -e "\nConfirmed: all input configs are valid."

outdir="$EXP_ROOT_DIR/$INSTANCE_NAME/$EXP_CONF_DIR"

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
