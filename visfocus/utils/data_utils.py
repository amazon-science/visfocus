import json
import logging
import random
import os
import braceexpand
import torch
from datasets.filesystems import S3FileSystem
import webdataset as wds
from typing import List
from webdataset.tariterators import tar_file_expander, base_plus_ext, valid_sample
import torchvision.transforms as transforms
from typing import Union, List, Tuple


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


# A copy from fm-astar/scripts/decoder-training/data/utils/transforms.py
def build_image_transform(image_size: Union[int, Tuple, List], is_train: bool = False):
    if type(image_size) == int:
        image_size = (image_size, image_size)
    elif type(image_size) == str:
        image_size = (int(image_size.split(',')[0]), int(image_size.split(',')[1]))
    if is_train:
        img_transforms = transforms.Compose([
            transforms.Resize(image_size),
            # ResizeAndPad(image_size),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.RandomChoice([
                transforms.RandomAffine(degrees=(2, 2)),
                # transforms.GaussianBlur((3,3)),
            ]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        img_transforms = transforms.Compose([
            transforms.Resize(image_size),
            # ResizeAndPad(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return img_transforms


def build_image_transform_p2s(cfg):
    from transformers import AutoProcessor
    patch_size = cfg['patch_size']
    cfg.update({'patch_size': {'height': patch_size, 'width': patch_size}})
    visual_preprocessor = AutoProcessor.from_pretrained(**cfg).image_processor

    return visual_preprocessor

def encode_ocr_with_bboxes(tokenizer, words, bboxes, add_special_tokens=True):
    ocr_tokens = tokenizer(words, add_special_tokens=add_special_tokens)
    word_tokens = []
    bboxes_tokens = []
    for index, token in enumerate(ocr_tokens.data['input_ids']):
        token_length = len(
            token) - 1  # since we used the tokenizer as a batch of words, each word has an extra EOS token
        bbox = torch.tensor(bboxes[index] * token_length).view(-1, 4)
        word_tokens.append(torch.tensor(token[:-1]))
        bboxes_tokens.append(bbox)
    word_tokens = torch.cat(word_tokens)
    bboxes_tokens = torch.cat(bboxes_tokens)
    return word_tokens, bboxes_tokens


def encode_ocr(tokenizer, words, add_special_tokens=True):
    ocr_tokens = tokenizer(words, add_special_tokens=add_special_tokens)
    word_tokens = []
    for index, token in enumerate(ocr_tokens.data['input_ids']):
        word_tokens.append(torch.tensor(token))
    word_tokens = torch.cat(word_tokens)
    return word_tokens


def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def get_ocr_info(json_file):
    ocr_info = []  # {'word' : transcription, 'bounding_box' : {topLeftX : , topLeftY : , width : , height :, rotation} }
    ocr_tokens = []  # list of words
    ocr_normalized_boxes = []  # list of bbox xyxy
    # with open(json_file) as f:
    #     data = json.load(f)
    data = json_file
    detections = data['TextDetections']
    for item in detections:
        if item['Type'] == 'WORD':
            text = item['DetectedText']
            ocr_tokens.append(text)
            bbox = item['Geometry']['BoundingBox']
            info = {'word': text,
                    'bounding_box':
                        {'topLeftX': bbox['Left'],
                         'topLeftY': bbox['Top'],
                         'width': bbox['Width'],
                         'height': bbox['Height']
                         }
                    }
            ocr_info.append(info)
            bbox_info = info['bounding_box']
            x = bbox_info.get("top_left_x", bbox_info["topLeftX"])
            y = bbox_info.get("top_left_y", bbox_info["topLeftY"])
            width = bbox_info["width"]
            height = bbox_info["height"]

            bbox_array = [x, y, x + width, y + height]
            ocr_normalized_boxes.append(bbox_array)
    return ocr_tokens, ocr_normalized_boxes


def verify_url(url: str):
    if not url.endswith("tar"):
        raise NotImplementedError("Only supports tar files")

    if not url.startswith("s3"):
        raise NotImplementedError("Only supports s3 streaming")


def s3fs_reader(s3fs_client: S3FileSystem, s3_urls: dict) -> dict:
    for s3_url in s3_urls:
        try:
            yield {"url": s3_url["url"],
                   "stream": s3fs_client.open(s3_url["url"].replace("s3://", ""), mode='rb')}
        except Exception as ex:
            logging.warning(ex)
            continue


def s3fs_to_stream(s3fs_client: S3FileSystem, src: dict) -> dict:
    streams = s3fs_reader(s3fs_client, src)
    files = tar_file_expander(streams, handler=wds.warn_and_continue)
    samples = group_by_keys_nothrow(files, handler=wds.warn_and_continue)
    return samples


def datasets_to_shards_list(datasets: List[str], root_dir: str) -> list:
    input_shards = [os.path.join(root_dir, ds) for ds in datasets]
    input_shards_list = []
    for url in input_shards:
        verify_url(url)
        input_shards_list.extend(braceexpand.braceexpand(url))
    if len(input_shards) > 1:
        random.shuffle(input_shards_list)
    return input_shards_list
