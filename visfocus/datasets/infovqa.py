import glob
import os

import numpy as np
import torch
from datasets import Dataset as HFDataset
from visfocus.utils.data_utils import load_json, get_ocr_info
from visfocus.data.vqa.vqa_dataset import VQADataItem
from transformers import ProcessorMixin
from tqdm import tqdm
import pandas as pd
from functools import reduce
import operator

from visfocus.utils.data_utils import encode_ocr_with_bboxes

if __name__ == '__main__':
    from docformer_v2.datasets import DATASETS_DIR
    from docformer_v2.datasets.base_dataset import BaseDataset
else:
    from . import DATASETS_DIR
    from .base_dataset import BaseDataset


class infographicsVQADataset(BaseDataset):
    TRAIN_PATHS = {'annotation_path': f'{DATASETS_DIR}/infographicsVQA/train_vf/annotations.json',
                   'ocr_path': 'datasets/infographicsVQA/train_vf/output.json',
                   'img_path': f'{DATASETS_DIR}/infographicsVQA/train_vf/png/',
                   'mode': 'train'}

    VAL_PATHS = {'annotation_path': f'{DATASETS_DIR}/infographicsVQA/val_vf/annotations.json',
                   'ocr_path': 'datasets/infographicsVQA/val_vf/output.json',
                   'img_path': f'{DATASETS_DIR}/infographicsVQA/val_vf/png/',
                   'mode': 'val'}

    TEST_PATHS = {'annotation_path': f'{DATASETS_DIR}/infographicsVQA/test_vf/annotations.json',
                   'ocr_path': 'datasets/infographicsVQA/test_vf/output.json',
                   'img_path': f'{DATASETS_DIR}/infographicsVQA/test_vf/png/',
                   'mode': 'test'}


    DATASET_ARGS = {'train': TRAIN_PATHS, 'val': VAL_PATHS, "test": TEST_PATHS}

    def __init__(self, dataset_args, tokenizer, mode='train', flush_cache=False,
                 visual_preprocessor: ProcessorMixin = None, **kwargs):
        self.dataset_path = f'{DATASETS_DIR}/infographicsVQA'
        super(infographicsVQADataset, self).__init__(dataset_args, tokenizer, mode=mode, flush_cache=False,
                                           visual_preprocessor=visual_preprocessor)
        self.detract_question_from_encoder = self.dataset_args.detract_question_from_encoder

    @classmethod
    def _load_dataset(cls, annotation_path, **kwargs):
        print('loading dataset')

        results = []
        ann = load_json(annotation_path)
        # ocr = load_json(ocr_path)
        skipped_assets = 0
        for samples in tqdm(ann):
            item = {}
            question = samples['question']
            if "answers" in samples.keys():
                answer = samples["answers"]
            else:
                answer = None
            item['answers'] = answer
            item['question'] = question
            item['image_id'] = samples['image_local_name'].split('.')[0]
            item['question_id'] = samples['question_id'] if 'question_id' in samples else samples['questionId' ]
            item['image'] = samples['image_path'] if 'image_path' in samples else f"{kwargs['img_path']}/{item['image_id']}.rectified.jpeg"
            # try:
            #     ocr_preds = ocr[samples['image'].split('/')[-1].split('.')[0]]
            # except:
            #     ocr_preds = ocr[samples['image'].split('/')[-1].split('.')[0][1:]]
            # if ocr_preds is not None:
            #     words, bboxes = get_ocr_info(ocr_preds)
            #     item['words'] = words
            #     item['bboxes'] = bboxes
            # else:
            #     skipped_assets += 1
            #     continue
            results.append(item)
        print(skipped_assets)
        return HFDataset.from_pandas(pd.DataFrame.from_records(results))

    @classmethod
    def load_dataset(cls, mode):
        if mode in cls.DATASET_ARGS:
            return cls._load_dataset(**cls.DATASET_ARGS[mode])

class InfoVQADataset_P2S(infographicsVQADataset):
    def __init__(self, *args, **kwargs):
        super(InfoVQADataset_P2S, self).__init__(*args, **kwargs)
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(kwargs['visual_preprocessor'])
        patch_size = self.processor.image_processor.patch_size['height'] * self.processor.image_processor.patch_size['width']
        self.max_patches = reduce(operator.mul, args[0].image_size) // patch_size
    
    def __getitem__(self, index):
        item = self.dataset[index]

        if item['image'].mode != 'RGB':
            item['image'] = item['image'].convert('RGB')
        
        image = item['image']
        questions = item['question']
        qid = item['question_id']
        # inputs = self.processor(text=questions, images=image, return_tensors="pt", add_special_tokens=False)
        inputs = self.processor(text=questions, images=image, return_tensors="pt", add_special_tokens=False, max_patches=self.max_patches)
        answer_list = item['answers']

        return {
            **inputs,
            **{'answer_list': answer_list,
               'questions': {'input_ids': torch.tensor(self.processor.tokenizer.encode(questions))},
               'question_id': qid}
        }

