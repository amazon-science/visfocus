import json
from collections import namedtuple

import deepspeed


class visfocusDeepspeed:
    __config = None
    __hf_config = None

    def __init__(self, *args, **kwargs):
        self.__init = None
        if self.__config and self.is_stage3_enabled():
            if "zero_optimization" in self.__config and "mics_shard_size" in self.__config["zero_optimization"]:
                self.__init = deepspeed.zero.MiCS_Init(config_dict_or_path=self.__hf_config.config, *args, **kwargs)
            else:
                self.__init = deepspeed.zero.Init(config_dict_or_path=self.__hf_config.config, *args, **kwargs)
    def __enter__(self):
        if self.__init:
            self.__init.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__init:
            self.__init.__exit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def load_state_dict(model, state_dict, *args, **kwargs):

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        def _load_state_dict(module, prefix=""):
            nonlocal state_dict, args, kwargs
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            sd = state_dict.copy()
            if metadata is not None:
                sd._metadata = metadata

            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            # because zero3 puts placeholders in model params, this context
            # manager gathers (unpartitions) the params of the current layer, then loads from
            # the state dict and then re-partitions them again
            with deepspeed.zero.GatheredParameters(
                    list(module.parameters(recurse=False)), modifier_rank=0
            ):
                module._load_from_state_dict(
                    state_dict=sd,
                    prefix=prefix,
                    local_metadata=local_metadata,
                    missing_keys=missing_keys,
                    unexpected_keys=unexpected_keys,
                    error_msgs=error_msgs,
                    *args,
                    **kwargs
                )

            for name, child in module._modules.items():
                if child is not None:
                    _load_state_dict(child, prefix + name + ".")

        _load_state_dict(model, "")
        return namedtuple('ret', 'missing_keys unexpected_keys')(missing_keys, unexpected_keys)

    @classmethod
    def initialize(cls, training_config):
        if getattr(training_config, "deepspeed", None) is not None:
            with open(training_config.deepspeed, 'r') as f:
                cls.__config = json.load(f)

            # Fills the 'auto' values
            from transformers.deepspeed import HfTrainerDeepSpeedConfig
            cls.__hf_config = HfTrainerDeepSpeedConfig(training_config.deepspeed)
            cls.__hf_config.trainer_config_process(training_config)
    @classmethod
    def is_stage3_enabled(cls):
        if cls.__config:
            if cls.__hf_config.config['zero_optimization']['stage'] == 3:
                return True
        return False
