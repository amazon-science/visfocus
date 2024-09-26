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


class AI2D(BaseDataset):
    TRAIN_PATHS = {'annotation_path': f'{DATASETS_DIR}/AI2D/ai2d/questions/train/',
                   'img_path': f'{DATASETS_DIR}/AI2D/ai2d/images/train/',
                   'mode': 'train'}

    VAL_PATHS = {'annotation_path': f'{DATASETS_DIR}/AI2D/ai2d/questions/test/',
                  'img_path': f'{DATASETS_DIR}/AI2D/ai2d/images/test/',
                  'mode': 'val'}

    TEST_PATHS = {'annotation_path': f'{DATASETS_DIR}/AI2D/ai2d/questions/test/',
                  'img_path': f'{DATASETS_DIR}/AI2D/ai2d/images/test/',
                  'mode': 'test'}

    DATASET_ARGS = {'train': TRAIN_PATHS, 'val': VAL_PATHS, "test": TEST_PATHS}

    def __init__(self, dataset_args, tokenizer, mode='train', flush_cache=False,
                 visual_preprocessor: ProcessorMixin = None, **kwargs):
        self.dataset_path = f'{DATASETS_DIR}/AI2D/ai2d'
        super(AI2D, self).__init__(dataset_args, tokenizer, mode=mode, flush_cache=flush_cache,
                                              visual_preprocessor=visual_preprocessor)
        self.detract_question_from_encoder = self.dataset_args.detract_question_from_encoder

    @classmethod
    def _load_dataset(cls, annotation_path, img_path, mode):
        print('loading dataset')
        results = []

        for annot_json in glob.glob(f'{annotation_path}*'):
            with open(annot_json) as f:
                img_annots = json.load(f)
            for q, ans in img_annots['questions'].items():
                item = {}
                item['question'] = f'{q} 1 {ans["answerTexts"][0]} 2 {ans["answerTexts"][1]} 3 {ans["answerTexts"][2]} 4 {ans["answerTexts"][3]}'
                item['answers'] = f'{ans["correctAnswer"] + 1}'
                item['image_id'] = img_annots['imageName'][:-4] # -> no extension needed
                item['question_id'] = img_annots['questions'][q]['questionId']
                item['image'] = os.path.join(img_path, img_annots['imageName'])
                results.append(item)
        return HFDataset.from_pandas(pd.DataFrame.from_records(results))

    @classmethod
    def load_dataset(cls, mode):
        if mode in cls.DATASET_ARGS:
            return cls._load_dataset(**cls.DATASET_ARGS[mode])
