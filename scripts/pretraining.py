import logging
import sys
import yaml

import transformers
from visfocus.data.base_module import BaseDataModule
from visfocus.data.pretraining.pretraining_module import PretrainingDataModule
from visfocus.data.pretraining.pretraining_wds_module import PretrainingWdsDataModule
from visfocus.engine.vqa_trainer import VQATrainer

from visfocus.utils.configuration import (DataArguments, ModelArguments,
                                       TrainingArguments)
from visfocus.utils.utils import print_params_info, load_checkpoint, get_model_from_string
from transformers import PreTrainedModel, Trainer


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model: PreTrainedModel = get_model_class(model_args)(model_args)

    print_params_info(model, logger)
    is_wds = 'wds' in data_args.data_config_path
    data_module_class = PretrainingWdsDataModule if is_wds else PretrainingDataModule
    data_module: BaseDataModule = data_module_class(
        input_tokenizer=model.input_tokenizer,
        output_tokenizer=model.output_tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    trainer: Trainer = VQATrainer(model=model, tokenizer=data_module.output_tokenizer, args=training_args,
                                  is_wds=is_wds,
                                  train_dataset=data_module.train_dataset, data_collator=data_module.data_collator)
    resume_from_checkpoint = load_checkpoint(training_args, logger)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def get_model_class(model_args):
    with open(model_args.model_config_path, 'rt') as f:
        model_config = yaml.safe_load(f)
    model_class = model_config.get('model_class', None)
    assert model_class is not None
    ret = get_model_from_string(model_class)
    logger.info(f"Extracted model_class: {model_class}, class: {ret}")
    return ret


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