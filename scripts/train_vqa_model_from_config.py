import logging
import os
import sys

import torch
import transformers
import yaml

import pathlib

# import wandb
assert os.environ['EXP_ROOT_DIR']

DEBUG_MODE = os.environ.get('DEBUG_MODE', None) == 'true'
TRAINING_STAGE = os.environ.get('TRAINING_STAGE', None)


def train():
    from transformers.integrations import TensorBoardCallback
    from transformers import PreTrainedModel, Trainer, is_torch_tpu_available
    from visfocus.data.base_module import BaseDataModule
    from visfocus.engine.vqa_trainer import VQATrainer, OCRTrainer, DocClsTrainer
    from visfocus.utils.configuration import (DataArguments, ModelArguments,
                                        TrainingArguments)
    from visfocus.utils.visfocus_deepspeed import visfocusDeepspeed
    from visfocus.utils.utils import print_params_info, load_checkpoint, \
        maybe_load_init_weights, get_model_class, setup_logger

    if not DEBUG_MODE:
        tensorboard_callback = TensorBoardCallback()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logger = setup_logger(model_args=model_args, data_args=data_args, training_args=training_args)

    if DEBUG_MODE:
        logger.warning('Running in debug mode.')

    visfocusDeepspeed.initialize(training_args)

    is_wds = 'wds' in data_args.data_config_path
    is_pretraining = 'pretrain' in data_args.data_config_path
    is_blueprint = 'blueprint' in data_args.data_config_path

    logger.info(f"Training mode - is_pretraining: {is_pretraining} wds: {is_wds}")
    
    if torch.distributed.is_initialized():
        num_gpus = torch.distributed.get_world_size()
    else:
        num_gpus = 1
    if is_wds and num_gpus > 1:
        training_args.per_device_eval_batch_size = num_gpus
        logger.info(f'Changing eval batch size to work with WebDatasets, new batch size : {num_gpus}')

    with visfocusDeepspeed():
        # model: PreTrainedModel = get_model_class(model_args, logger)(model_args)
        model: PreTrainedModel = get_model_class(model_args, logger)
        model.to("cuda")
    model_arch = getattr(model, 'model_arch', 'vqa_former')
    model = maybe_load_init_weights(model, logger)

    print_params_info(model, logger)
    input_tokenizer = model.input_tokenizer
    output_tokenizer = model.output_tokenizer


    data_module_class = get_data_module_class(is_pretraining, is_wds, is_blueprint=is_blueprint)
    data_module: BaseDataModule = data_module_class(
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        data_args=data_args,
        training_args=training_args,
        visual_model_name=model.vision_model.model_name if hasattr(model, 'vision_model') and model.vision_model is not None else None,
        model_arch=model_arch,
    )
    trainer_class = OCRTrainer if data_module.task == 'donut_pretrain' else VQATrainer
    
    if training_args.label_names is not None:
        training_args.label_names = training_args.label_names[0].split(',')

    trainer: Trainer = trainer_class(model=model, tokenizer=data_module.input_tokenizer, args=training_args,
                                     is_wds=is_wds,
                                     **data_module.to_dict(do_train=training_args.do_train),
                                     compute_metrics=None, # this is because the score is computed from files
                                     out_dir=training_args.run_name,
                                     )
    resume_from_checkpoint = load_checkpoint(training_args, logger)
    if training_args.do_train:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if training_args.do_eval:
        if resume_from_checkpoint and os.path.isdir(training_args.last_checkpoint):
            logger.info(f"Loading checkpoint {training_args.last_checkpoint}")
            trainer._load_from_checkpoint(training_args.last_checkpoint)
        trainer.evaluate(eval_dataset=data_module.eval_dataset)
        node = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if node == 0:
            trainer.aggregate_metric_fn(ckpt_path=getattr(training_args, 'last_checkpoint', ''))


def get_data_module_class(is_pretraining, is_wds, is_blueprint=False):
    from visfocus.data.pretraining.pretraining_module import PretrainingDataModule
    from visfocus.data.pretraining.pretraining_wds_module import PretrainingWdsDataModule
    from visfocus.data.vqa.vqa_module import VQAFormerDataModule
    from visfocus.data.vqa.vqa_wds_module import VQAFormerWdsDataModule

    if is_pretraining:
        data_module_class = PretrainingWdsDataModule if is_wds else PretrainingDataModule
    else:
        data_module_class = VQAFormerWdsDataModule if is_wds else VQAFormerDataModule
    return data_module_class


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    train()
