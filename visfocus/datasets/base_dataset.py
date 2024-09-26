import numpy as np
import os
import logging
import torch
from datasets import Dataset as HFDataset, Image
from torch.utils.data import Dataset

from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.pix2struct.image_processing_pix2struct import Pix2StructImageProcessor

from visfocus.utils.data_utils import encode_ocr_with_bboxes, encode_ocr
from visfocus.utils.data_utils import build_image_transform
from visfocus.data.vqa.vqa_dataset import VQADataItem


logger = logging.getLogger(__name__)


class BaseDataset(Dataset):

    def __init__(self, dataset_args, tokenizer, mode='train', flush_cache=False,
                 visual_preprocessor: ProcessorMixin = None):
        self.dataset_args = dataset_args
        self.mode = mode
        self.tokenizer = tokenizer
        self.vis_processor = build_image_transform(self.dataset_args.image_size, is_train=(self.mode == 'train'))
        self.flush_cache = True # flush_cache
        self.setup()
        self.dataset = self.load(mode)
        if visual_preprocessor is not None:
            self.dataset = self.dataset.cast_column('image', Image())

        self.detract_question_from_encoder = self.dataset_args.detract_question_from_encoder
        if self.detract_question_from_encoder:
            logging.info('Detracting the question from the encoder input')

    def setup(self):
        # function to download data from s3 to desired folderpath if no such folders exist
        pass

    @classmethod
    def load_dataset(cls, mode):
        pass

    def load(self, mode, dataset_mapper_func=None):
        # json_dataset_path = os.path.join(self.dataset_path, f'hf_dataset_{mode}_sample.jsonl') ############
        json_dataset_path = os.path.join(self.dataset_path, f'hf_dataset_{mode}.jsonl') ############
        if os.path.exists(json_dataset_path) and not self.flush_cache:
            dataset = HFDataset.from_json(json_dataset_path)
            if dataset_mapper_func is not None:
                dataset = dataset.map(dataset_mapper_func, num_proc=int(20))
        else:
            dataset = self.load_dataset(mode)
        save_to_disk = True
        if save_to_disk and (not os.path.exists(json_dataset_path) or self.flush_cache):
            print('Saving to disk')
            dataset.to_json(json_dataset_path)
            print(f"Dataset saved : {json_dataset_path}")
        return dataset

    def __getitem__(self, index):
        item = self.dataset[index]

        # ocr_words = item['words']
        # ocr_bbox = item['bboxes']
        if item['image'].mode != 'RGB':
            item['image'] = item['image'].convert('RGB')
        if isinstance(self.vis_processor, Pix2StructImageProcessor):
            image = self.vis_processor(item['image'], return_tensors='pt')
            image.data = image.flattened_patches[0]
        else:
            image = self.vis_processor(item['image'])
        # words = ocr_words + [self.tokenizer.eos_token]
        # bboxes = ocr_bbox + [[0, 0, 0, 0]]
        # if not self.detract_question_from_encoder:
            # words = [item['question'] + [':']] + words
            # bboxes = [[0, 0, 1, 1] + [1, 1, 1, 1]] + bboxes
        # input_ids = encode_ocr(self.tokenizer, words, add_special_tokens=False)
        # token_boxes = (token_boxes * 1000).type(torch.int)
        # input_mask = [1] * len(input_ids)
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

    def __len__(self):
        return len(self.dataset)