import os
import sys
import datetime
import logging
import shutil
import random
import collections
import copy

import numpy as np
import torch


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id, fs_prefix='./'):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(fs_prefix, 'output', exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def set_gpu(gpu):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        return


class ConfigNode(collections.OrderedDict):
    IMMUTABLE = "__is_frozen"

    def __init__(self, init_dict={}):
        self.__dict__[ConfigNode.IMMUTABLE] = False
        super().__init__(init_dict)

        for key in self:
            if isinstance(self[key], collections.abc.Mapping):
                self[key] = ConfigNode(self[key])
            elif isinstance(self[key], list):
                for idx, item in enumerate(self[key]):
                    if isinstance(item, collections.abc.Mapping):
                        self[key][idx] = ConfigNode(item)

    def freeze(self):
        for field in self.keys():
            if isinstance(self[field], collections.abc.Mapping):
                self[field].freeze()
            elif isinstance(self[field], list):
                for item in self[field]:
                    if isinstance(item, collections.abc.Mapping):
                        item.freeze()

        self.__dict__[ConfigNode.IMMUTABLE] = True

    def defrost(self):
        for field in self.keys():
            if isinstance(self[field], collections.abc.Mapping):
                self[field].defrost()
            elif isinstance(self[field], list):
                for item in self[field]:
                    if isinstance(item, collections.abc.Mapping):
                        item.defrost()

        self.__dict__[ConfigNode.IMMUTABLE] = False

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)

        return self[key]

    def __setattr__(self, key, value):
        if self.__dict__[ConfigNode.IMMUTABLE] is True:
            raise AttributeError("ConfigNode has been frozen and can't be updated")

        self[key] = value

    def _indent(self, st, num_spaces):
        st = st.split("\n")
        first = st.pop(0)
        st = [(num_spaces * " ") + line for line in st]
        st = [first] + st
        st = "\n".join(st)
        return st

    def __str__(self):
        strs = []

        if isinstance(self, collections.abc.Mapping):
            for key, value in sorted(self.items()):
                seperator = "\n" if isinstance(value, ConfigNode) else " "
                if isinstance(value, list):
                    attr_str = ["{}:".format(key)]
                    for item in value:
                        item_str = self._indent(str(item), 2)
                        attr_str.append("- {}".format(item_str))
                    attr_str = "\n".join(attr_str)
                else:
                    attr_str = "{}:{}{}".format(str(key), seperator, str(value))
                    attr_str = self._indent(attr_str, 2)
                strs.append(attr_str)
        return "\n".join(strs)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super().__repr__())


def save_checkpoint(model, tokenizer, configs, output_dir, logger):
    logger.info("Saving model checkpoint to %s", output_dir)
    model = copy.deepcopy(model)
    if hasattr(model, "vision_model"):
        vision_model = model.vision_model
        torch.save(
            {"model": vision_model.state_dict()},
            os.path.join(output_dir, "vision_model.pth")
        )
        del model.vision_model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(configs, os.path.join(output_dir, "training_configs.bin"))
    logger.info("Saved model checkpoint to %s", output_dir)
