import logging
import os
import yaml

from visfocus.utils.utils import Dict2Object


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
logger = logging.getLogger()


class BaseModel(object):

    @staticmethod
    def add_extra_tokens(model, tokenizer, extra_token_file):
        logger.info(f'adding tokens from {extra_token_file}')
        with open(extra_token_file, 'r') as f:
            extra_tokens = f.read().splitlines()

        token_size_before = len(tokenizer)
        tokenizer.add_tokens(extra_tokens)
        model.resize_token_embeddings(len(tokenizer))
        token_size_after = len(tokenizer)

        logger.info(f'added {token_size_after - token_size_before} tokens')

    @staticmethod
    def get_config_from_model_args(model_args):
        config = None
        if hasattr(model_args, 'model_config_path'):
            # Load the YAML file into a dictionary
            with open(model_args.model_config_path, "r") as f:
                yaml_content = yaml.safe_load(f)
                config = Dict2Object(yaml_content)
                logger.info(f"Model config:\n{yaml_content}")
        # Populate args to config
        if hasattr(model_args, 'vqa_model_path') and model_args.vqa_model_path is not None:
            config.vqa_former.vqa_model_path = model_args.vqa_model_path
        if hasattr(model_args, 'llm_model_name_or_path') and model_args.llm_model_name_or_path is not None:
            config.language_model.model_name_or_path = model_args.llm_model_name_or_path
        if hasattr(model_args, 'model_init_weights') and model_args.model_init_weights is not None:
            config.model_init_weights = model_args.model_init_weights
        if hasattr(model_args, 'weights_names_prefix_to_freeze') and model_args.weights_names_prefix_to_freeze is not None:
            weights_names_prefix_to_freeze = model_args.weights_names_prefix_to_freeze.split(",")
            config.language_model.weights_names_prefix_to_freeze = weights_names_prefix_to_freeze
        return config

    def _get_config_from_model_args(self, model_args):
        self.model_args = model_args
        config = self.get_config_from_model_args(model_args)
        self.yaml_config = config
        return config

    def freeze_weights(self, weights_names_prefix_to_freeze):
        params_freezed_names = []
        params_trainable_names = []
        n_params_freezed = 0

        def _should_freeze(name):
            return any(name.startswith(p) for p in weights_names_prefix_to_freeze)

        for name, param in self.named_parameters():
            if _should_freeze(name):
                param.requires_grad = False
            is_frozen = param.requires_grad is False
            if is_frozen:
                params_freezed_names.append(name)
                n_params_freezed += param.numel()
            else:
                params_trainable_names.append(name)
        if 'language_model' in weights_names_prefix_to_freeze:
            self.language_model = self.language_model.eval()
        logger.info(f"Num parameters freezed: {n_params_freezed}")
        logger.info(f"Frozen weights: {params_freezed_names}")
        logger.info(f"Trainable weights: {params_trainable_names}")

    def save_configs(self, save_directory):
        attrs = ['yaml_config']
        for attr in attrs:
            path_to_config = os.path.join(save_directory, attr + '.yaml')
            with open(path_to_config, 'w') as f:
                yaml.dump(getattr(self, attr), f)

    def load_configs(self, save_directory):
        logger.info(f"loading config from: {save_directory}")
        attrs = ['yaml_config']
        for attr in attrs:
            path_to_config = os.path.join(save_directory, attr + '.yaml')
            with open(path_to_config, 'r') as f:
                attr_config = yaml.safe_load(f, Loader=yaml.Loader)
                setattr(self, attr, attr_config)

