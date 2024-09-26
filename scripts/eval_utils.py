import copy
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer

from multipagevqa.models.load_model import load_model
from multipagevqa.utils.data import get_tokenizer
from multipagevqa.utils.evaluation import NLS, EvalAIAnswerProcessor
from tqdm import tqdm
import json
from time import time
from multipagevqa.utils.visualization import plot_doc
import os
from multipagevqa.utils.data import split_input
from multipagevqa.utils.global_variables import *
import logging
# from multipagevqa.trainers.DocVQA_trainer import DocVQATrainer
from multipagevqa.utils.data import get_tokenizer, get_collate_fn
from joblib import Parallel, delayed
from functools import partial

logger = logging.getLogger(__name__)


def eval_DocVQA(model, configs, dataloader, device, **kwargs):
    eval_idx = 0
    extract_answer_fn = get_answer_fn(configs)
    model.to(device)
    # model.eval()
    logger.info('*' * 50)
    logger.info('Eval num: {} model path: {}'.format(eval_idx, configs.model.hf_model))
    list_dict_ans = []
    N_examples = 0
    pbar = tqdm(dataloader)
    for idx, inputs in enumerate(pbar):
        inputs, val_inputs = split_input(inputs)
        page_window = configs.general.prediction_page_window
        input_pages = inputs['document_pages'][0].shape[-1]
        if input_pages > page_window:
            for start_page in range(input_pages - page_window):
                input_window = break_input_into_pages(copy.copy(inputs), start_page, page_window)
                dict_answer = predict(input_window, val_inputs, model, extract_answer_fn)
                dict_answer['label_id'] = val_inputs['gt_answer'].detach().cpu().numpy().tolist()
                list_dict_ans += [dict_answer]
        else:
            dict_answer = predict(inputs, val_inputs, model, extract_answer_fn)
            dict_answer['gt_answer'] = val_inputs['gt_answer'].detach().cpu().numpy().tolist()
            list_dict_ans.append(dict_answer)
        dict_answer
        N_examples += 1

    return list_dict_ans


def predict(inputs, val_inputs, model, extract_answer_fn):
    M, N = inputs['decoder_input_ids'].size()
    new_decoder_input_ids = torch.full((M, N + 1), model.config.decoder_start_token_id)
    new_decoder_input_ids[..., :N] = inputs['decoder_input_ids']
    inputs['decoder_input_ids'] = new_decoder_input_ids

    new_decoder_attention_mask = torch.full((M, N + 1), 1)
    new_decoder_attention_mask[..., :N] = inputs['decoder_attention_mask']
    inputs['decoder_attention_mask'] = new_decoder_attention_mask

    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    model_output = model.generate(**inputs, max_length=100)
    batch_predicted_answer = extract_answer_fn(model_output[:, inputs['decoder_input_ids'].shape[1]:])
    batch_question = extract_answer_fn(inputs['decoder_input_ids'])
    answers = val_inputs[ANSWER_NAME]

    for i in range(len(batch_predicted_answer)):
        dict_answer = {}
        dict_answer['nls'] = None

        if ANSWER_NAME in val_inputs.keys():
            dict_answer['predicted_answer'] = batch_predicted_answer[i]
            dict_answer['nls'] = NLS(val_inputs[ANSWER_NAME][i], batch_predicted_answer[i])

        for key in [QUESTION_ID_NAME, PAGE_NUMBER_NAME]:
            if key in val_inputs.keys():
                dict_answer[key] = val_inputs[key][i]

        # number of pages
        dict_answer['number_of_pages'] = val_inputs[NUMBER_OF_PAGES_NAME][i][0]

        # add number of pages
        dict_answer['question'] = batch_question[i]
        dict_answer['gt_answer'] = answers[i]
        dict_answer['gt_answer_page'] = val_inputs['answer_page_idx'][i][0]
        dict_answer[DOC_ID_NAME] = val_inputs[DOC_ID_NAME][i]
    return dict_answer


def make_input_windows(inputs, page_window):
    input_list = []
    for key in inputs:
        item = inputs[key]


def break_input_into_pages(inputs, start_page, window_length, **kargs):
    chunks_idx = list(torch.where((inputs['page_number'] == -100).any(dim=0))[0])
    start_idx = chunks_idx[start_page]
    end_idx = chunks_idx[start_page + window_length]
    for key in inputs:
        if 'label' not in key and 'decoder' not in key and key != 'relative_answer_page_idx' and key != 'document_pages' and key != 'doc_images':
            assert (inputs[key][:, start_idx] == -100).all() and (inputs[key][:, end_idx] == -100).all()

            item = inputs[key][:, start_idx: end_idx + 1]
            inputs[key] = item
        elif key == 'document_pages' or key == 'doc_images':
            item = inputs[key][:, start_page: start_page + window_length]
            inputs[key] = item

    return inputs


def get_answer_fn(configs):
    if configs.data.dataset_name in [MP_DOC_VQA_NAME, DUDE_NAME]:
        answer_fn = get_T5_answer(configs)
    else:
        raise Exception('Do not provide this model answers: {}'.format(configs.model.model_name))

    return answer_fn


class get_T5_answer():
    def __init__(self, configs):
        self.tokenizer = get_tokenizer(configs)

    def __call__(self, model_output, **kwargs):
        prediction = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        return prediction


def save_file_for_submission(list_dict_ans, configs):
    list_dict_submit = []
    for i in list_dict_ans:
        if MP_DOC_VQA_NAME in configs.data.dataset_name:
            dict_sample = {
                'questionId': i['questionId'],
                'answer': i['predicted_answer'],
                'answer_page': 1
            }
        elif DUDE_NAME in configs.data.dataset_name:
            dict_sample = {
                'questionId': i['questionId'],
                'answer': i['predicted_answer'],
                'answer_confidence': 1.0,
                'answer_abstain': False
            }

        list_dict_submit.append(dict_sample)
    model_checkpoint = configs.model.hf_model.split('/')[-1]
    results_path = os.path.join(
        configs.train.output_dir,
        'submission_{}_{}_res.json'.format(
            configs.data.dataset_split,
            model_checkpoint
        )
    )

    if configs.general.debug_mode == False:
        with open(results_path, 'w') as f:
            json.dump(list_dict_submit, f)


def save_file_for_analysis(list_dict_ans, number_of_samples, max_number_of_pages, configs):
    results_path = os.path.join(
        configs.train.output_dir,
        'Mistakes_analysis_{}.json'.format(number_of_samples)
    )

    if configs.general.debug_mode == False:
        dict_analysis = {}
        list_anls = []
        for i in range(len(list_dict_ans)):
            list_anls.append(list_dict_ans[i]['nls'])
        dict_analysis['anls'] = np.mean(list_anls)

        # save all examples
        dict_analysis['metadata'] = list_dict_ans

        list_anls = [[] for i in range(max_number_of_pages)]
        list_anls_per_page_number = [[[] for i in range(max_number_of_pages)] for i in range(max_number_of_pages)]
        for i in range(len(list_dict_ans)):
            number_of_pages = list_dict_ans[i]['number_of_pages'] - 1

            list_anls[number_of_pages].append(list_dict_ans[i]['nls'])
            answer_page = 0
            if list_dict_ans[i]['gt_answer_page'] is not None:
                answer_page = list_dict_ans[i]['gt_answer_page']
            list_anls_per_page_number[number_of_pages][answer_page].append(
                list_dict_ans[i]['nls'])

        for i in range(max_number_of_pages):
            dict_analysis['anls for {} pages. Number of samples {}'.format(i + 1, len(list_anls[i]))] = np.mean(
                list_anls[i])

        for number_of_pages, list_page_nls in enumerate(list_anls_per_page_number):
            for answer_in_page, list_nlp_per_pages in enumerate(list_page_nls):
                dict_analysis[
                    'anls for {} pages where answer appear in page {}. Number of samples {}'.format(number_of_pages + 1,
                                                                                                    answer_in_page + 1,
                                                                                                    len(list_nlp_per_pages))
                ] = np.mean(list_nlp_per_pages)

        for number_of_pages in range(max_number_of_pages):
            list_specific_page_dict_mistakes = []
            for i in range(len(list_dict_ans)):
                dict_mistake = {}
                if (list_dict_ans[i]['number_of_pages'] == number_of_pages + 1) and (
                        list_dict_ans[i]['nls'] < configs.general.nls_mistake_th):
                    dict_mistake['question'] = list_dict_ans[i]['question']
                    dict_mistake[DOC_ID_NAME] = list_dict_ans[i][DOC_ID_NAME]
                    dict_mistake['nls'] = list_dict_ans[i]['nls']
                    dict_mistake['answer prediction'] = list_dict_ans[i]['predicted_answer']
                    dict_mistake['GT answer'] = list_dict_ans[i]['gt_answer']
                    dict_mistake[QUESTION_ID_NAME] = list_dict_ans[i][QUESTION_ID_NAME]

                    list_specific_page_dict_mistakes.append(dict_mistake)

            dict_analysis['{} pages mistakes'.format(number_of_pages + 1)] = list_specific_page_dict_mistakes

        with open(results_path, 'w') as f:
            json.dump(dict_analysis, f, indent=4)

