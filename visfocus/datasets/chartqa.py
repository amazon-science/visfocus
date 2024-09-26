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

from PIL import Image
from transformers.models.pix2struct.image_processing_pix2struct import render_header

from visfocus.utils.data_utils import encode_ocr_with_bboxes

if __name__ == '__main__':
    from docformer_v2.datasets import DATASETS_DIR
    from docformer_v2.datasets.base_dataset import BaseDataset
else:
    from . import DATASETS_DIR
    from .base_dataset import BaseDataset


class ChartQADataset(BaseDataset):
    TRAIN_PATHS = {'annotation_path': {
                    'human': f'{DATASETS_DIR}/ChartQA/train/train_human.json',
                    'augmented': f'{DATASETS_DIR}/ChartQA/train/train_augmented.json',
                },
                   'ocr_path': 'datasets/ChartQA/train/output.json',
                   'img_path': f'{DATASETS_DIR}/ChartQA/train/png/',
                   'mode': 'train'}

    # VAL_PATHS = {'annotation_path': {
    #                 'human': f'{DATASETS_DIR}/ChartQA/val/val_human.json',
    #                 'augmented': f'{DATASETS_DIR}/ChartQA/val/val_augmented.json',
    #             },
    #                'ocr_path': 'datasets/ChartQA/val/output.json',
    #                'img_path': f'{DATASETS_DIR}/ChartQA/val/png/',
    #                'mode': 'val'}

    VAL_PATHS = {'annotation_path': {
                    'human': f'{DATASETS_DIR}/ChartQA/test/test_human.json',
                    'augmented': f'{DATASETS_DIR}/ChartQA/test/test_augmented.json',
                },

                 'ocr_path': 'datasets/ChartQA/test/output.json',
                 'img_path': f'{DATASETS_DIR}/ChartQA/test/png/',
                 'mode': 'test'}
    
    TEST_PATHS = {'annotation_path': {
                    'human': f'{DATASETS_DIR}/ChartQA/test/test_human.json',
                    'augmented': f'{DATASETS_DIR}/ChartQA/test/test_augmented.json',
                },

                 'ocr_path': 'datasets/ChartQA/test/output.json',
                 'img_path': f'{DATASETS_DIR}/ChartQA/test/png/',
                 'mode': 'test'}

    DATASET_ARGS = {'train': TRAIN_PATHS, 'val': VAL_PATHS, "test": TEST_PATHS}

    global _subset # human / augmented

    def __init__(self, dataset_args, tokenizer, mode='train', flush_cache=False,
                 visual_preprocessor: ProcessorMixin = None, prompted = None,semi_prompted = None, **kwargs):
        self.dataset_path = f'{DATASETS_DIR}/ChartQA'
        
        self.train_joint_splits = True # merge human and augmented splits
        ChartQADataset._subset = dataset_args.name.split('_')[1] if mode != 'train' and self.train_joint_splits else None
        
        super(ChartQADataset, self).__init__(dataset_args, tokenizer, mode=mode, flush_cache=flush_cache,
                                           visual_preprocessor=visual_preprocessor)
        self.detract_question_from_encoder = self.dataset_args.detract_question_from_encoder

    @classmethod
    def _load_dataset(cls, annotation_path, ocr_path, img_path, mode):
        print('loading dataset')

        results = []
        annots = [load_json(annotation_path[subset]) for subset in ['human', 'augmented']] if not cls._subset and mode == 'train' else [load_json(annotation_path[cls._subset])]
        # ocr = load_json(ocr_path)
        skipped_assets = 0
        for ann in annots:
            for samples in tqdm(ann):
                item = {}
                question = samples['query']
                if "label" in samples.keys():
                    answer = samples["label"]
                else:
                    answer = None
                    print('None')
                item['answers'] = [answer]
                item['question'] = question
                item['image_id'] = samples['imgname']
                item['question_id'] = samples['imgname']
                item['image'] = img_path + samples['imgname']
                # try:
                #     ocr_preds = ocr[samples['imgname'].split('/')[-1].split('.')[0]]
                # except:
                #     ocr_preds = ocr[samples['imgname'].split('/')[-1].split('.')[0][1:]]
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


class ChartQADataset_P2S(ChartQADataset):
    def __getitem__(self, index):
        item = self.dataset[index]

        if item['image'].mode != 'RGB':
            item['image'] = item['image'].convert('RGB')

        item['image'] = Image.fromarray(render_header(item['image'], header=item['question']))

        image = self.vis_processor(item['image'])
        if self.mode == 'train':
            values, counts = np.unique(item['answers'], return_counts=True)
            answer = np.random.choice(values, p=counts / counts.sum())
        else:
            answer = item['answers']

        return VQADataItem(
            input_ids=None, # input_ids,
            attention_mask=None, # torch.tensor(input_mask, dtype=torch.int8),
            token_boxes=None,
            lm_labels=None,
            visual_feature=None,
            pm_labels=None,
            segment_ids=None,
            # for LLM tokenizer:
            question=item['question'],
            answer=answer,
            # for evaluation:
            sample_unique_id_index=item['question_id'],
            doc_unique_id_index=item['image_id'],
            question_index_in_pool=None,
            is_eval=self.mode != 'train',
            image=image
        )


from functools import reduce
import operator


class ChartQADataset_Eval_P2S(ChartQADataset):
    def __init__(self, *args, **kwargs):
        super(ChartQADataset_Eval_P2S, self).__init__(*args, **kwargs)

        from transformers import AutoProcessor
        visual_preprocessor = AutoProcessor.from_pretrained(kwargs['visual_preprocessor'])

        self.processor = visual_preprocessor
        patch_size = self.processor.image_processor.patch_size['height'] * self.processor.image_processor.patch_size['width']
        self.max_patches = reduce(operator.mul, args[0].image_size) // patch_size

    def __getitem__(self, index):
        item = self.dataset[index]

        if item['image'].mode != 'RGB':
            item['image'] = item['image'].convert('RGB')
        
        image = item['image']
        questions = item['question']
        # inputs = self.processor(text=questions, images=image, return_tensors="pt", add_special_tokens=False)
        inputs = self.processor(text=questions, images=image, return_tensors="pt", add_special_tokens=False, max_patches=self.max_patches)
        answer_list = item['answers']
        qid = item['question_id']

        return {
            **inputs,
            **{'answer_list': answer_list,
               'questions': {'input_ids': torch.tensor(self.processor.tokenizer.encode(questions))},
               'question_id': qid}
        }
