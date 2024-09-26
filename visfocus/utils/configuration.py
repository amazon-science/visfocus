from dataclasses import dataclass, field
from typing import Optional, List

import transformers


@dataclass
class ModelArguments:
    model_config_path: str = field(default="configs/configs_ofir/pretraining_wds_model.yaml",
                                   metadata={"help": "Path to the model config."})
    vqa_model_path: str = field(default=None,
                                metadata={"help": "Path to the VQA pretrained model. Default as in model config."})
    llm_model_name_or_path: str = field(default=None,
                                        metadata={"help": "Path to the language model. Default as in model config."})
    model_init_weights: str = field(default=None,
                                        metadata={"help": "Path to models weights to start training from."})
    weights_names_prefix_to_freeze: str = field(default=None,
                                        metadata={"help": "Weights name prefix to freeze"})
    freeze_modules: Optional[List[str]] = field(default=None)


@dataclass
class DataArguments:
    data_config_path: str = field(default="configs/configs_ofir/pretraining_wds_data.yaml",
                                  metadata={"help": "Path to the data config."})
    data_root_dir: str = field(default=None,
                               metadata={"help": "Path to the data root dir. Default as in data config."})
    smart_order: bool = field(default=None)
    data_max_seq_length: int = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_config_path: str = field(default="configs/configs_ofir/pretraining_wds_training.yaml",
                                   metadata={"help": "Path to the training config."})
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    evaluation_strategy: str = field(default="steps")
    num_train_epochs: float = field(default=8.0)
    max_steps: int = field(default=1000000)
    log_level: str = field(default="info")
    bf16: bool = field(default=False)
    eval_steps: int = field(default=5000)
    logging_steps: int = field(default=50)
    logging_nan_inf_filter: bool = field(default=False)
    include_inputs_for_metrics: bool = field(default=False)
    warmup_steps: int = field(default=4200)
    anls_iou_threshold: float = field(default=0.0)
    load_last_last_checkpoint: bool = field(default=False)

