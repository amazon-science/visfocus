import glob
import os
import json

import numpy as np
import torch
from datasets import Dataset as HFDataset
from transformers import ProcessorMixin
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

from visfocus.utils.data_utils import encode_ocr_with_bboxes

if __name__ == '__main__':
    from visfocus.datasets import DATASETS_DIR
    from visfocus.datasets.base_dataset import BaseDataset
else:
    from . import DATASETS_DIR
    from .base_dataset import BaseDataset


class CORD(BaseDataset):
    def __init__(self, dataset_args, tokenizer, mode='train', flush_cache=False,
                 visual_preprocessor: ProcessorMixin = None, **kwargs):
        self.dataset_path = f'{DATASETS_DIR}/CORD'
        super(CORD, self).__init__(dataset_args, tokenizer, mode=mode, flush_cache=flush_cache,
                                              visual_preprocessor=visual_preprocessor)
        self.detract_question_from_encoder = self.dataset_args.detract_question_from_encoder
    
    def load(self, mode):
        dataset = self.load_dataset(mode)
        return dataset
    
    @classmethod
    def _load_dataset(cls, mode):
        print('loading dataset')
        mode = 'validation' if mode == 'val' else mode
        return load_dataset("naver-clova-ix/cord-v1", split=mode)

    @classmethod
    def load_dataset(cls, mode):
        return cls._load_dataset(mode=mode)
