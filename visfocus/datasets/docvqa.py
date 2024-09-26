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


class DocVQADataset(BaseDataset):
    TRAIN_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/train/train_v1.0.json',
                   'ocr_path': f'{DATASETS_DIR}/DocVQA/train/output.json',
                   'img_path': f'{DATASETS_DIR}/DocVQA/train/',
                   'mode': 'train'}

    VAL_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_v1.0.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val'}

    VAL_DENSE_400_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_dense_400.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_DENSE_500_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_dense_500.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_DENSE_600_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_dense_600.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_DENSE_800_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_dense_800.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_DENSE_1000_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_dense_1000.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_SPARSE_500_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_sparse_500.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_SPARSE_400_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_sparse_400.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_SPARSE_300_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_sparse_300.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_SPARSE_200_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_sparse_200.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_SPARSE_100_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_sparse_100.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_SPARSE_50_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_sparse_50.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

# non-cummulative
    VAL_800_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_800-.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_650_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_650-.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_400_800_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_400-800.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_200_400_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_200-400.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_0_200_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_0-200.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_200_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_200-.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_300_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_300-.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}


    VAL_0_400_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_0-400.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_400_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_400-.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_0_500_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_0-500.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_500_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_500-.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_900_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_900-.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_700_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_700-.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    VAL_600_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/val/val_600-.json',
                 'ocr_path': f'{DATASETS_DIR}/DocVQA/val/output.json',
                 'img_path': f'{DATASETS_DIR}/DocVQA/val/',
                 'mode': 'val_dense_v1'}

    # test actually..
    # VAL_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/test/test_v1.0_with_gt.json',
    #               'ocr_path': f'{DATASETS_DIR}/DocVQA/test/output.json',
    #               'img_path': f'{DATASETS_DIR}/DocVQA/test/',
    #               'mode': 'test'}

    TEST_PATHS = {'annotation_path': f'{DATASETS_DIR}/DocVQA/test/test_v1.0.json',
                  'ocr_path': f'{DATASETS_DIR}/DocVQA/test/output.json',
                  'img_path': f'{DATASETS_DIR}/DocVQA/test/',
                  'mode': 'test'}

    DATASET_ARGS = {'train': TRAIN_PATHS, 'val': VAL_PATHS, "test": TEST_PATHS, 
                    'val_dense_400': VAL_DENSE_400_PATHS,
                    'val_dense_500': VAL_DENSE_500_PATHS,
                    'val_dense_600': VAL_DENSE_600_PATHS,
                    'val_dense_800': VAL_DENSE_800_PATHS,
                    'val_dense_1000': VAL_DENSE_1000_PATHS,
                    'val_sparse_500': VAL_SPARSE_500_PATHS,
                    'val_sparse_400': VAL_SPARSE_400_PATHS,
                    'val_sparse_300': VAL_SPARSE_300_PATHS,
                    'val_sparse_200': VAL_SPARSE_200_PATHS,
                    'val_sparse_100': VAL_SPARSE_100_PATHS,
                    'val_sparse_50': VAL_SPARSE_50_PATHS,
                    'VAL_400_800_PATHS': VAL_400_800_PATHS,
                    'VAL_200_400_PATHS': VAL_200_400_PATHS,
                    'VAL_0_400_PATHS': VAL_0_400_PATHS,
                    'VAL_0_500_PATHS': VAL_0_500_PATHS,
                    'VAL_0_200_PATHS': VAL_0_200_PATHS,
                    'VAL_900_PATHS': VAL_900_PATHS,
                    'VAL_800_PATHS': VAL_800_PATHS,
                    'VAL_700_PATHS': VAL_700_PATHS,
                    'VAL_650_PATHS': VAL_650_PATHS,
                    'VAL_600_PATHS': VAL_600_PATHS,
                    'VAL_500_PATHS': VAL_500_PATHS,
                    'VAL_400_PATHS': VAL_400_PATHS,
                    'VAL_300_PATHS': VAL_300_PATHS,
                    'VAL_200_PATHS': VAL_200_PATHS,
                    }

    def __init__(self, dataset_args, tokenizer, mode='train', flush_cache=False,
                 visual_preprocessor: ProcessorMixin = None, **kwargs):
        self.dataset_path = f'{DATASETS_DIR}/DocVQA'
        super(DocVQADataset, self).__init__(dataset_args, tokenizer, mode=mode, flush_cache=flush_cache,
                                              visual_preprocessor=visual_preprocessor)
        self.detract_question_from_encoder = self.dataset_args.detract_question_from_encoder

    @classmethod
    def _load_dataset(cls, annotation_path, ocr_path, img_path, mode):
        print('loading dataset')

        results = []
        ann = load_json(annotation_path)['data']
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
            item['image_id'] = f'{samples["ucsf_document_id"]}_{samples["ucsf_document_page_no"]}'
            item['question_id'] = samples['questionId']
            item['image'] = img_path + samples['image']
            # try:
            #     ocr_preds = ocr[samples['image'].split('/')[-1].split('.')[0]]
            # except:
            #     ocr_preds = ocr[samples['image'].split('/')[-1].split('.')[0][1:]]
            # if ocr_preds is not None:
            #     words, bboxes = get_ocr_info(ocr_preds)
            #     item['words'] = words
                # item['bboxes'] = bboxes
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


class DocVQADataset_P2S(DocVQADataset):
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
