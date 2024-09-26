import glob
import os
import json

import numpy as np
import torch
from datasets import Dataset as HFDataset
from visfocus.utils.data_utils import load_json, get_ocr_info
from visfocus.data.vqa.vqa_dataset import VQADataItem
from transformers import ProcessorMixin
from tqdm import tqdm
import pandas as pd

from visfocus.utils.data_utils import encode_ocr_with_bboxes

if __name__ == '__main__':
    from docformer_v2.datasets import DATASETS_DIR
    from docformer_v2.datasets.base_dataset import BaseDataset
else:
    from . import DATASETS_DIR
    from .base_dataset import BaseDataset


class OCRVQA(BaseDataset):
    TRAIN_PATHS = {'annotation_path': f'{DATASETS_DIR}/OCR-VQA/train.json',
                   'img_path': f'{DATASETS_DIR}/OCR-VQA/images/',
                   'mode': 'train'}

    VAL_PATHS = {'annotation_path': f'{DATASETS_DIR}/OCR-VQA/test.json',
                 'img_path': f'{DATASETS_DIR}/OCR-VQA/images/',
                 'mode': 'val'}

    TEST_PATHS = {'annotation_path': f'{DATASETS_DIR}/OCR-VQA/test.json',
                  'img_path': f'{DATASETS_DIR}/OCR-VQA/images/',
                  'mode': 'test'}

    DATASET_ARGS = {'train': TRAIN_PATHS, 'val': VAL_PATHS, "test": TEST_PATHS}

    def __init__(self, dataset_args, tokenizer, mode='train', flush_cache=False,
                 visual_preprocessor: ProcessorMixin = None, **kwargs):
        self.dataset_path = f'{DATASETS_DIR}/OCR-VQA'
        super(OCRVQA, self).__init__(dataset_args, tokenizer, mode=mode, flush_cache=flush_cache,
                                              visual_preprocessor=visual_preprocessor)
        self.detract_question_from_encoder = self.dataset_args.detract_question_from_encoder

    @classmethod
    def _load_dataset(cls, annotation_path, img_path, mode):
        print('loading dataset')
        results = []

        with open(annotation_path) as f:
            annots = json.load(f)

        for annot in annots:
            for i, (q, a) in enumerate(zip(annot['questions'], annot['answers'])):
                item = {}
                item['question'] = q
                item['answers'] = [a]
                item['image_id'] = os.path.basename(annot['imageURL']).split('.')[0]
                item['question_id'] = f'{item["image_id"]}-{i}'
                item['image'] = os.path.join(img_path, os.path.basename(annot['imageURL']))
                results.append(item)
        return HFDataset.from_pandas(pd.DataFrame.from_records(results))

    @classmethod
    def load_dataset(cls, mode):
        if mode in cls.DATASET_ARGS:
            return cls._load_dataset(**cls.DATASET_ARGS[mode])
