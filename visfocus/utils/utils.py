import importlib
import sys
from typing import Union
from pathlib import Path
import os
import yaml
import logging
import torch
from pprint import pformat
import transformers
from transformers.trainer_utils import get_last_checkpoint
import importlib


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


class Dict2Object:
    def __init__(self, input_dict: dict):
        my_dict = input_dict.copy()
        for k, v in input_dict.items():
            if isinstance(v, dict):
                my_dict[k] = Dict2Object(v)

        self._my_dict = my_dict

    def __getattr__(self, name):
        try:
            return self._my_dict[name]
        except KeyError:
            raise AttributeError(f"'MyObject' object has no attribute '{name}'")

    def __getitem__(self, item):
        return getattr(self, item)

    def __setstate__(self, state):
        my_dict = state['_my_dict']
        for k, v in state.items():
            if isinstance(v, dict):
                my_dict[k] = Dict2Object(v)
        self._my_dict = my_dict
        return my_dict

    def __str__(self):
        return str(self._my_dict)

    def to_dict(self):
        return self._my_dict.copy()

def get_torch_dtype(precision: Union[str, int]) -> torch.dtype:
    """
    Convert precision to torch dtype
    """
    if precision in [16, '16', 'fp16']:
        return torch.float16
    elif precision in ["bf16"]:
        return torch.bfloat16
    elif precision in [32, '32']:
        return torch.float32
    else:
        raise NotImplementedError(f"precision {precision} not implemented")


def string_split_by_start_end(string, start, end):
    """
    Args:
        string (str): string
        start (str): start token
        end (str): end token

    Returns:
        List[str]: All the substrings of string that starts with start and ends with end
    """
    if len(string) == 0: return []
    ret = string.split(start)

    # if string starts with start, the the first part is empty,
    # otherwise the first part does not starts with start and we want to drop it
    ret = ret[1:]

    ret = [x[:-len(end)] for x in ret if x.endswith(end)]
    return ret


def string_split_multiple_delimiters(string, delimiters):
    """

    Args:
        string (str): _description_
        delimiters (Union[List[str], str]): list of delimiters

    Returns:
        List[str]: substrings of string, split by all the delimiters
    """
    if isinstance(delimiters, str):
        delimiters = [delimiters]
    base_delim = delimiters[0]
    for delim in delimiters[1:]:
        string = string.replace(delim, base_delim)
    return string.split(base_delim)


def num_non_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def num_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_params_info(model, logger):
    n_trainable = num_trainable_parameters(model)
    n_frozen = num_non_trainable_parameters(model)
    logger.info(f"Initialized model -- num trainable paramters: {n_trainable}, "
                f"num frozen parameters: {n_frozen} (total {n_trainable + n_frozen})")


def load_checkpoint(training_args, logger):
    resume_from_checkpoint = False
    if training_args.load_last_last_checkpoint:
        last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) \
            else None
        if last_checkpoint is not None:
            logger.info(f"Found checkpoint in output_dir ({training_args.output_dir}), "
                        f"resuming from checkpoint: {last_checkpoint}")
            resume_from_checkpoint = True
        else:
            logger.info(f"Checkpoint not found in output_dir ({training_args.output_dir})")
        training_args.last_checkpoint = last_checkpoint
    if (not resume_from_checkpoint) and training_args.resume_from_checkpoint:
        logger.info(f"Loading resume from checkpoint {training_args.resume_from_checkpoint}")
        resume_from_checkpoint = training_args.resume_from_checkpoint

    return resume_from_checkpoint


def adjust_state_dict(state_dict,model_state_dict):
    unwrapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module") or key.startswith("_forward_module"):  # deepspeed zero_to_fp32.py
            new_key = ".".join(key.split(".")[2:])
        elif key.startswith("model"):
            new_key = ".".join(key.split(".")[1:])
        else:
            # execution path if loading pytorch_model.bin
            # key does not need to change
            new_key = key

        unwrapped_state_dict[new_key] = value
    if 'lm_head.weight' not in unwrapped_state_dict.keys() and 'lm_head.weight' in model_state_dict:
        if 'transformer.wte.weight' in unwrapped_state_dict:
            unwrapped_state_dict['lm_head.weight'] = unwrapped_state_dict['transformer.wte.weight']
        else:
            unwrapped_state_dict['lm_head.weight'] = unwrapped_state_dict['shared.weight']
    return unwrapped_state_dict


def maybe_load_init_weights(model, logger):
    try:
        path_to_weights = getattr(model.yaml_config, 'model_init_weights', None)
        if path_to_weights and os.path.isfile(path_to_weights):
            logger.info(f"Loading weights from {path_to_weights}")
            state_dict = torch.load(path_to_weights)
            current_state_dict = model.state_dict()
            if model.model_arch == 'blip2':
                ret = model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loading result: {ret}")
                return model
            state_dict = adjust_state_dict(state_dict, current_state_dict)
            ret = model.load_state_dict(state_dict, strict = False)
            if len(ret.unexpected_keys) > 0:
                raise Exception(f"{ret}")
            for n in ret.missing_keys:
                if current_state_dict[n].requires_grad:
                    raise Exception(f"{ret}")
            logger.info(f"Loading result: {ret}")
    except Exception as e:
        logger.info(f"Failed to load init weights: {e}")
    return model

def get_model_from_string(model_path: str):
    """
    Given a string of the model location, returns the model
    For example:
        get_model_string("visfocus.vqa_former_model.VQAFormerModel")
    returns VQAFormerModel
    """
    parts = model_path.split(".")
    model_path, model_name = ".".join(parts[:-1]), parts[-1]
    pkg = importlib.import_module(model_path)
    return getattr(pkg, model_name)


def setup_logger(**kwargs):
    # Setup logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if 'training_args' in kwargs:
        print("create dir ",kwargs['training_args'].output_dir)
        Path(kwargs['training_args'].output_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(kwargs['training_args'].output_dir, 'log.txt')))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers,
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    for k, v in kwargs.items():
        if k == 'training_args':
            continue
        logger.info(f"{k}: {pformat(v)}")

    return logger

def pad_sequence_with_side(x, side, **kwargs):
    batch_first = kwargs.get('batch_first', False)
    if side == 'left':
        x = [xx.flip([0]) for xx in x]

    x = torch.nn.utils.rnn.pad_sequence(
        x,
        **kwargs,
    )

    if side == 'left':
        x = x.flip(dims=[1 if batch_first else 0])

    return x

def get_model_class(model_args, logger=None, last_ckpt=False):
    with open(model_args.model_config_path, 'rt') as f:
        model_config = yaml.safe_load(f)
    model_class = model_config.get('model_class', None)
    # if logger is not None:
    #     ret = get_model_from_string(model_class)
    #     logger.info(f"Extracted model_class: {model_class}, class: {ret}")
        
    assert model_class.split('.')[-1].startswith('VisFocus')
    ret = get_visfocus_model(model_args, model_class, logger, last_ckpt=last_ckpt)
    return ret


def get_visfocus_model(args, model_class, logger, last_ckpt=False):
    from visfocus.models.vf_models import VisFocusConfig as VFConf
    from visfocus.models import T5Tokenizer, bert_encoder
    from visfocus.utils.general import ConfigNode

    model_class = model_class.split('.')
    lib, model = '.'.join(model_class[:-1]), model_class[-1]
    lib = importlib.import_module(lib)
    model = getattr(lib, model)
    
    with open(args.model_config_path, "r") as f:
        config = ConfigNode(yaml.safe_load(f)).visfocus
        
    if "vision" in config.model:
        with open(config.model.vision, "r") as f:
            config.model.vision = ConfigNode(yaml.safe_load(f))
            config.model.vision.model.image_size = config.image_size
    else:
        config.model.vision = None

    model_config = VFConf.from_pretrained(config.model.variant.replace('vf', 't5'), cache_dir=None)
    
    model_config.freeze_modules = getattr(config, 'freeze_modules', [])
    model_config.unfreeze_modules = getattr(config, 'unfreeze_modules', [])
    model_config.vision = config.model.get("vision", None)
    model_config.vqa_method = config.get("vqa_method", None)
    model_config.matcher_type = config.get("matcher_type", 'default')
    model_config.lora = config.get("lora", None)
    model_config.vl_l1_loss = config.get("vl_l1_loss", None)
    model_config.generate_max_new_tokens_len = config.generate_max_new_tokens_len
    tokenizer = T5Tokenizer.from_pretrained(
        config.model.variant.replace('vf', 't5'),
        do_lower_case=config.do_lower_case,
        cache_dir=None,
        model_max_length=config.max_seq_length
    )
    tokenizer.generate_max_new_tokens_len = model_config.generate_max_new_tokens_len
    
    model = model.from_pretrained(
            config.model_name_or_path if not last_ckpt else last_ckpt,
            from_tf=bool(".ckpt" in config.model_name_or_path),
            config=model_config,
            cache_dir=None,
            logger=logger,
            ignore_mismatched_sizes=True,
        )
    model.input_tokenizer = model.output_tokenizer =  model.language_model_tokenizer = tokenizer

    try:
        if isinstance(model.text_embedder, bert_encoder.BERT):
            # if text embedder is not the same as the LM encdoer, we need to convert token ids using original tokenizer
            model.text_embedder.orig_tokenizer = model.input_tokenizer
    except:
        logger.warn('no text_embedder found (assuming LtR is operating...)')

    model.add_task_tokens()
    if model.model_arch == 'ocrf_for_doc_cls':
        model.add_cls_tokens('rvl_cdip')
    if hasattr(config, 'task_name'):
        model.set_task_name(config.task_name)
    else:
        logger.info(f"WARNING: task_name was not specified. Using model class default ({model.task_name}).")
    
    logger.info("Loaded model successfully from %s", config.model_name_or_path)
    return model


def get_falcon_model(args, model_class, logger, last_ckpt=False):
    from visfocus.models.falcon_dfm import OCRFreeFalconConfig as VFConf
    from visfocus.models import AutoTokenizer
    from visfocus.utils.general import ConfigNode

    model_class = model_class.split('.')
    lib, model = '.'.join(model_class[:-1]), model_class[-1]
    lib = importlib.import_module(lib)
    model = getattr(lib, model)
    
    with open(args.model_config_path, "r") as f:
        config = ConfigNode(yaml.safe_load(f)).falcon
        
    if "vision" in config.model:
        with open(config.model.vision, "r") as f:
            config.model.vision = ConfigNode(yaml.safe_load(f))
            config.model.vision.model.image_size = config.image_size
    else:
        config.model.vision = None

    model_config = VFConf.from_pretrained(config.model_name_or_path, cache_dir=None)
    
    model_config.freeze_modules = getattr(config, 'freeze_modules', [])
    model_config.unfreeze_modules = getattr(config, 'unfreeze_modules', [])
    model_config.vision = config.model.get("vision", None)
    model_config.vqa_method = config.get("vqa_method", None)
    model_config.matcher_type = config.get("matcher_type", 'default')
    model_config.generate_max_new_tokens_len = config.generate_max_new_tokens_len
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        do_lower_case=config.do_lower_case,
        cache_dir=None,
        model_max_length=config.max_seq_length,
        use_fast=False
    )
    tokenizer.generate_max_new_tokens_len = model_config.generate_max_new_tokens_len
    
    model = model.from_pretrained(
            config.model_name_or_path if not last_ckpt else last_ckpt,
            from_tf=bool(".ckpt" in config.model_name_or_path),
            config=model_config,
            cache_dir=None,
            logger=logger,
            ignore_mismatched_sizes=True
        )
    model.input_tokenizer = model.output_tokenizer =  model.language_model_tokenizer = tokenizer
    
    if getattr(model.input_tokenizer, 'pad_token_id') is None:
        model.input_tokenizer.add_special_tokens({"pad_token": '|<padtoken>|', 'bos_token': '<|beginningoftext|>'})
        # model.input_tokenizer.pad_token_id = model.input_tokenizer.eos_token_id

    model.generation_config.bos_token_id = model.input_tokenizer.bos_token_id
    model.generation_config.eos_token_id = model.input_tokenizer.eos_token_id
    model.generation_config.pad_token_id = model.input_tokenizer.pad_token_id

    if hasattr(config, 'task_name'):
        task = getattr(config, 'task_name', None)
        # if task == 'ocr_mpm' and torch.distributed.is_initialized():
        #     tasks = task.split('_')
        #     task = tasks[torch.distributed.get_rank() % 2]
        model.set_task_name(task)
    model.add_task_tokens()
    logger.info("Task Name: %s", model.task_name)

    logger.info("Loaded model successfully from %s", config.model_name_or_path)
    return model


def get_pix2struct_model(cfg):
    from transformers import AutoProcessor
    from visfocus.models.pix2struct import Pix2StructForConditionalGeneration
    
    model_name = cfg['model_class'].split('-')
    if len(model_name) == 3:
        # if model is finetuned
        arch, ft_dataset, version = model_name
        model_signature = f'google/{arch}-{ft_dataset}-{version}'
    elif len(model_name) == 4:
        # if model is finetuned
        arch, a1, a2, version = model_name
        model_signature = f'google/{arch}-{a1}-{a2}-{version}'
    elif len(model_name) == 2:
        arch, version = model_name
        model_signature = f'google/{arch}-{version}'

    model = Pix2StructForConditionalGeneration.from_pretrained(model_signature)
    model.model_arch = cfg['model_arch']
    processor = AutoProcessor.from_pretrained(model_signature)

    tokenizer = processor.tokenizer
    tokenizer.model_max_length = 512
    tokenizer.generate_max_new_tokens_len = cfg['generate_max_new_tokens_len']
    model.input_tokenizer = model.output_tokenizer = model.language_model_tokenizer = tokenizer
    
    model.vision_model = Dict2Object({'model_name': model_signature})
    
    return model

def get_donut_model(cfg):
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    # from visfocus.models.pix2struct import Pix2StructForConditionalGeneration
    
    model_name = f'naver-clova-ix/{cfg["model_class"]}'

    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.model_arch = cfg['model_arch']
    processor = DonutProcessor.from_pretrained(model_name)

    tokenizer = processor.tokenizer
    tokenizer.model_max_length = 512
    tokenizer.generate_max_new_tokens_len = cfg['generate_max_new_tokens_len']
    model.input_tokenizer = model.output_tokenizer = model.language_model_tokenizer = tokenizer
    
    model.vision_model = Dict2Object({'model_name': model_name})
    
    return model


def get_t5_wrapper_model(logger=None):
    from visfocus.models.t5_wrapper import T5_wrapper

    lora_def_conf = {
        'task_type': 'CAUSAL_LM',
        'r': 16,
        'lora_alpha': 32,
        'bias': 'none',
        'lora_dropout': 0.01,
        'target_modules': ['q', 'v']
        }
    return T5_wrapper(t5_variant='small',
                        lora_conf=lora_def_conf,
                        freeze=False,
                        logger=logger)