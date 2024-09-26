
import copy
import torch
import numpy as np
from tqdm import tqdm
import json
import os
import logging
from joblib import Parallel, delayed
from functools import partial
logger = logging.getLogger(__name__)


def eval_model(model, dataloader, configs, **kwargs):
    eval_DocVQA_fn = partial(
        eval_DocVQA,
        model,
        configs
    )
    if type(dataloader) is not list:
        dataloader = [dataloader]
    n_jobs = len(dataloader)
    print(torch.cuda.device_count())
    available_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())][:n_jobs]

    if n_jobs>len(available_devices):
        # available_devices += ['cpu']*(n_jobs-len(available_devices))
        available_devices = ['cpu']*n_jobs


    assert n_jobs == len(available_devices)
    max_number_of_pages = dataloader[0].dataset.samples_loader.max_number_pages
    if not configs.general.debug_mode:
        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_DocVQA_fn)(dataloader[i], available_devices[i]) for i in range(n_jobs))
    else:
        results = [eval_DocVQA_fn(dataloader[0], available_devices[0])]
    list_nls = []
    list_dict_ans = []
    for result in results:
        for sample in result:
            list_nls.append(sample['nls'])
            list_dict_ans.append(sample)

    N_examples = len(list_nls)
    logger.info("\tIdx: {}, ANLS {}".format(N_examples, np.mean(list_nls)))
    # save_file_for_submission(list_dict_ans, configs)
    # save_file_for_analysis(list_dict_ans, number_of_samples=N_examples, max_number_of_pages=max_number_of_pages,
    #                        configs=configs)
    model_checkpoint = configs.model.hf_model.split('/')[-1]
    logger.info("\tThe results for model {} are {}".format(model_checkpoint, np.mean(list_nls)))

