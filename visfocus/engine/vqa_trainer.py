import logging
import random
import re
from typing import Dict, Union, Any, Optional, List, Tuple
import os

import numpy as np
import torch
from visfocus.metrics.scorers.fscorer import FScorer
from torch import nn
from transformers.trainer import Trainer
import webdataset as wds

from visfocus.metrics.anls import ANLSMetric
from visfocus.metrics.base_metric import BaseMetric
from visfocus.metrics.vqa_metrics import VQAMetrics
from visfocus.metrics.classification_metrics import ClassificationMetrics

from tqdm import tqdm
from visfocus.metrics.scorers.anls_scorer import AnlsScorer
import torch.distributed as dist
from transformers.generation import GenerationConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from transformers.utils import is_sagemaker_mp_enabled

from visfocus.models.vf_models import VisFocus_OCR_MPM
from transformers import VisionEncoderDecoderModel, DonutProcessor
import re


logger = logging.getLogger()

PROMPT_DENOISING_PATTERN = r'(?P<key><extra_id_\d*>)(?P<value>.*?)(?:<sep/>|$)'


class VQATrainer(Trainer):

    def __init__(self, out_dir=None, **kwargs):
        self.is_wds = kwargs.pop('is_wds', False)
        super().__init__(**kwargs)

        self.compute_metrics = kwargs.get('compute_metrics', VQAMetrics(tokenizer=self.tokenizer))
        self.is_test = kwargs['args'].do_predict
        logdir_name = 'test_logs' if self.is_test else 'metric_logs'
        self.out_dir_metrics = os.path.join(out_dir, logdir_name)
        os.makedirs(self.out_dir_metrics, exist_ok=True)

        anls_iou_threshold = getattr(self.args, "anls_iou_threshold", 0.0)
        self.anls_metric = ANLSMetric(iou_threshold=anls_iou_threshold)
        self.generate_max_length = self.model.language_model_tokenizer.generate_max_new_tokens_len
        self.generation_config = self.model.generation_config

        if 'chartqa' in self.out_dir_metrics:
            # relaxed accuracy
            self.aggregate_metric_fn = self.run_chartqa_ra_eval
        elif 'textvqa' in self.out_dir_metrics or 'tat' in self.out_dir_metrics or 'ai2d' in self.out_dir_metrics or 'ocrvqa' in self.out_dir_metrics:
            # accuracy
            self.aggregate_metric_fn = self.run_textvqa_acc_eval
        else:
            # ANLS
            self.aggregate_metric_fn = self.aggregate_predictions

    def _truncate_padded_input(self, inputs):
        padding_side = self.tokenizer.padding_side
        if 'attention_mask' not in inputs:
            return inputs
        non_zero = torch.nonzero(inputs["attention_mask"].sum(0), as_tuple=True)
        if len(non_zero[0] > 0):
            if padding_side == 'left':
                first_nonzero_idx = non_zero[0][0].item()
                if first_nonzero_idx % 4 != 0:
                    first_nonzero_idx = (first_nonzero_idx // 4) * 4
                for k, v in inputs.items():
                    if "image" not in k and isinstance(v[0], torch.Tensor):
                        inputs[k] = v[:, first_nonzero_idx:].contiguous()
            else:
                last_nonzero_idx = non_zero[0][-1].item() + 1
                if last_nonzero_idx % 4 != 0:
                    last_nonzero_idx = (1 + last_nonzero_idx // 4) * 4
                for k, v in inputs.items():
                    if "image" not in k and isinstance(v[0], torch.Tensor):
                        inputs[k] = v[:, :last_nonzero_idx].contiguous()
        return inputs

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.is_wds and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            node = torch.distributed.get_rank()
            size = torch.distributed.get_world_size()
            inputs = {k: v[node::size] for k, v in inputs.items()}
            inputs = self._truncate_padded_input(inputs)

        inputs = self._prepare_inputs(inputs)
        vqa_former_visual_feature = inputs['image'] if 'image' in inputs else None

        # if self.model.vision_model:
        #     vqa_former_visual_feature = self.model.vision_model(inputs['pixel_values'])
        if isinstance(model, VisionEncoderDecoderModel):
            model_inputs = {k: v for k, v in inputs.items() if k in {'pixel_values', 'decoder_input_ids'}}
            model_inputs.update({
                # 'max_length': model.decoder.config.max_position_embeddings,
                'pad_token_id': self.model.input_tokenizer.pad_token_id,
                'eos_token_id': self.model.input_tokenizer.eos_token_id,
                'use_cache': True,
                'bad_words_ids': [[self.model.input_tokenizer.unk_token_id]],
                # 'return_dict_in_generate': True,

            })
        else:
            model_inputs = {k: v for k, v in inputs.items() if k in {'questions', 'answers', 'image', 'labels'}}
        outputs = getattr(model, 'module', model).generate(**model_inputs, max_new_tokens=self.generate_max_length, generation_config=self.generation_config)
        # outputs = getattr(model , 'module', model).generate(**model_inputs,
        #                                             do_sample=False,
        #                                             top_p=0.9,
        #                                             temperature=1,
        #                                             num_beams=5,
        #                                             max_new_tokens=self.generate_max_length,
        #                                             min_length=1,
        #                                             repetition_penalty=1.5,
        #                                             length_penalty=1.0,
        #                                             num_return_sequences=1,
        #                                             generation_config=self.generation_config)

        prediction_ids = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

        if isinstance(model, VisionEncoderDecoderModel):
            sequences = self.eval_dataset.vis_processor.batch_decode(prediction_ids)
            predictions = []
            for s in sequences:
                s = s.replace(self.eval_dataset.vis_processor.tokenizer.eos_token, "").replace(self.eval_dataset.vis_processor.tokenizer.pad_token, "")
                s = re.sub(r"<.*?>", "", s, count=1).strip()  # remove first task start token
                try:
                    predictions.append(self.eval_dataset.vis_processor.token2json(s)['answer'])
                except KeyError:
                    predictions.append('NONE')

        else:
            model_arch = getattr(model, 'model_arch', 'vqa_former')
            if model_arch == 'decoder_only':
                prediction_ids = prediction_ids[:, model_inputs['input_ids'].shape[1]:]
            predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)

        node = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if not self.is_test:
            # if not test and labels are available
            losses, closet_answers = self.calculate_losses_and_closest_answer(inputs, predictions)

            loss = torch.tensor(losses, device=prediction_ids.device, dtype=torch.float32)

            labels = self.tokenizer.batch_encode_plus(
                closet_answers,
                padding="longest",
                truncation=True,
                max_length=self.generate_max_length,
                return_tensors="pt",
            ).input_ids.to(prediction_ids.device)

            # logging some answers
            # only in gpu i for some prop. p
            if node == 0:
                if random.random() > .9:
                    str_to_print = ''
                    _keys = ['input_ids']
                    input_ids = [inputs['questions'][k] for k in _keys][0]
                    prompt = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                    for i in range(len(predictions)):
                        str_to_print += f"\nprompt: {prompt[i]} \n pred: {predictions[i]} \n labels: {inputs['answer_list'][i]} \n score : {losses[i]} \n"
                    logger.info(str_to_print)
        else:
            # test_mode, dump preds
            closet_answers = ['none'] * len(predictions)
            labels = loss = torch.tensor(0.).repeat(len(predictions)).cuda()

        # eval data is local
        if not model.training:
            write_mode = getattr(self, f'write_mode_{node}', 'w')
            image_ids, question_ids = inputs.get('image_id', ['none']*len(loss)), inputs.get('question_id', ['none']*len(loss))
            for i, s in enumerate(loss):
                image_id, question_id = image_ids[i], question_ids[i]
                with open(f'{self.out_dir_metrics}/error_logs_rank{node}.txt', write_mode, encoding="utf-8") as f:
                    q = self.tokenizer.decode(inputs['questions']['input_ids'][i], skip_special_tokens=True)
                    if self.is_test:
                        f.write('{"answer": "' + str(predictions[i]).replace('"', "'") + '", ' + '"questionId": ' + str(question_id) + '}\n')
                    else:
                        f.write(f"{str(image_id).strip()}@@{question_id}@@{q}@@{closet_answers[i]}@@{predictions[i]}@@{s}\n")
                write_mode = 'a' # after writing the first line, from now on append
            setattr(self, f'write_mode_{node}', 'a')

        return (loss, prediction_ids, labels)

    def calculate_losses_and_closest_answer(self, inputs, predictions):
        scorer = AnlsScorer()
        preds = scorer.from_items(predictions)
        annots = scorer.from_items(inputs["answer_list"])
        scorer.add(preds, annots)
        losses = scorer.scores
        indices = scorer.indices
        closet_answers = [ans[indices[index]] for index, ans in enumerate(inputs["answer_list"])]
        return losses, closet_answers

    def _get_wds_loader(self, split, **kwargs):

        if split == 'train':
            dataset = self.train_dataset
            data_collator = self.data_collator
            batch_size = self._train_batch_size
        elif split == 'eval':
            eval_dataset = kwargs.pop("eval_dataset", None)
            dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            batch_size = self.args.eval_batch_size
        else:
            raise NotImplementedError(f"{split}")

        _kwargs = {}
        if self.args.dataloader_num_workers > 0:
            logger.warning("dataloader_num_workers > 0, set multiprocessing_context to spawn")
            _kwargs['multiprocessing_context'] = 'spawn'
        loader = wds.WebLoader(
            dataset, batch_size=None, shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            **_kwargs,
        )

        return loader

    def get_train_dataloader(self):
        if self.is_wds:
            return self._get_wds_loader(split='train')
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self.is_wds:
            return self._get_wds_loader(split='eval', eval_dataset=eval_dataset)
        else:
            return super().get_eval_dataloader()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["global_step"] = self.state.global_step
            logs["total_steps"] = self.state.max_steps

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        from transformers.utils import ADAPTER_CONFIG_NAME, ADAPTER_SAFE_WEIGHTS_NAME, ADAPTER_WEIGHTS_NAME, CONFIG_NAME, SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME
        from transformers.configuration_utils import PretrainedConfig
        from transformers.trainer import __version__

        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)

        if not any(
            os.path.isfile(f)
            for f in [
                weights_file,
                safe_weights_file,
                weights_index_file,
                safe_weights_index_file,
                adapter_weights_file,
                adapter_safe_weights_file,
            ]
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
                        )
                    state_dict = torch.load(weights_file, map_location="cpu")
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                load_fsdp_model(self.accelerator.state.fsdp_plugin, self.accelerator, model, resume_from_checkpoint)
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    state_dict = torch.load(weights_file, map_location="cpu")

                vision_sd = {}
                lm_sd = {}
                for k,v in state_dict.items():
                    if k.startswith('vision_model'):
                        vision_sd[k[len('vision_model.'):]] = v
                    else:
                        lm_sd[k] = v

                load_result = model.load_state_dict(lm_sd, strict=False)
                missing_keys, unexpected_keys = load_result
                missing_keys = [k for k in missing_keys if not k.startswith('vision_model')]
                logger.warning(f'missing_keys: {missing_keys}')
                logger.warning(f'unexpected_keys: {unexpected_keys}')
                logger.info("Language model and Projection MLP loaded successfully")

                self.load_vision_pretrained(model.vision_model, vision_sd, logger)

                # release memory
                del state_dict
                del lm_sd
                del vision_sd
                # self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif is_peft_available() and isinstance(model, PeftModel):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint):
                    model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    @staticmethod
    def load_vision_pretrained(model, state_dict, logger, model_type='swin'):

        if model_type == "swin":
            # delete relative_position_index since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete relative_coords_table since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete attn_mask since we always re-init it
            attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
            for k in attn_mask_keys:
                del state_dict[k]

            # bicubic interpolate relative_position_bias_table if not match
            relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
            for k in relative_position_bias_table_keys:
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = model.state_dict()[k]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    logger.warning(f"Error in loading {k}, passing......")
                else:
                    if L1 != L2:
                        # bicubic interpolate relative_position_bias_table if not match
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                            mode='bicubic')
                        state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

            # bicubic interpolate absolute_pos_embed if not match
            absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
            for k in absolute_pos_embed_keys:
                # dpe
                absolute_pos_embed_pretrained = state_dict[k]
                absolute_pos_embed_current = model.state_dict()[k]
                _, L1, C1 = absolute_pos_embed_pretrained.size()
                _, L2, C2 = absolute_pos_embed_current.size()
                if C1 != C1:
                    logger.warning(f"Error in loading {k}, passing......")
                else:
                    if L1 != L2:
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                        absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                        absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                            absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                        absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                        absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                        state_dict[k] = absolute_pos_embed_pretrained_resized

            # bicubic interpolate absolute_pos_embed if not match
            # pattern = re.compile(r'vl_cross_attn_layers.*positional_embedding')
            pattern = re.compile(r'.*cross_attn.*positional_embedding')
            absolute_pos_embed_keys = [s for s in state_dict.keys() if pattern.match(s)]
            for k in absolute_pos_embed_keys:
                # dpe
                absolute_pos_embed_pretrained = state_dict[k]
                absolute_pos_embed_current = model.state_dict()[k]
                L1, C1 = absolute_pos_embed_pretrained.size()
                L2, C2 = absolute_pos_embed_current.size()
                if C1 != C1:
                    logger.warning(f"Error in loading {k}, passing......")
                else:
                    if L1 != L2:
                        logger.warning(f'Interpolating positional embedding (source: {L1}, dest: {L2})')
                        # absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                        absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(1, 0).unsqueeze(0)
                        absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                            absolute_pos_embed_pretrained, size=L2, mode='linear')
                        absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 1).squeeze(0)
                        state_dict[k] = absolute_pos_embed_pretrained_resized

        # import pdb;pdb.set_trace()
        msg = model.load_state_dict(state_dict, strict=False)
        missing_keys, unexpected_keys = msg
        missing_keys = [k for k in missing_keys if 'relative_coords' not in k and 'relative_position' not in k]
        logger.warning(f'missing_keys: {missing_keys}')
        logger.warning(f'unexpected_keys: {unexpected_keys}')

        logger.info("Vision model loaded successfully")
        # torch.cuda.empty_cache()

    def aggregate_predictions(self, ckpt_path=None):
        nodewise_log_file = [f'{self.out_dir_metrics}/{fn}' for fn in os.listdir(self.out_dir_metrics)]
        preds = {}
        for log_file in nodewise_log_file:
            with open(os.path.join(log_file), 'r', encoding='utf-8') as f:
                for l in f.readlines():
                    _, qid, _, _, _, score =  l.strip().split('@@')
                    preds[qid] = float(score)
        scores = list(preds.values())
        logger.info(f'\nFinal Score (ANLS): {round(sum(scores) / len(scores), 3)}\n\tover {len(scores)} examples.\nckpt_path: {ckpt_path}')

    def run_chartqa_ra_eval(self, ckpt_path=None):
        from visfocus.metrics.vqa_metrics import relaxed_correctness

        nodewise_log_file = [f'{self.out_dir_metrics}/{fn}' for fn in os.listdir(self.out_dir_metrics)]
        preds = []
        for log_file in nodewise_log_file:
            with open(os.path.join(log_file), 'r') as f:
                preds += [(l.split('@@')[-3], l.split('@@')[-2]) for l in f.readlines()]
        scores = []
        for t in preds:
            scores.append(relaxed_correctness(*t))
        logger.info(f'\nFinal Score (Relaxed Accuracy): {round(sum(scores) / len(scores), 3)}\n\tover {len(scores)} examples.\nckpt_path: {ckpt_path}')

    def run_textvqa_acc_eval(self, ckpt_path=None):
        nodewise_log_file = [f'{self.out_dir_metrics}/{fn}' for fn in os.listdir(self.out_dir_metrics)]
        preds = []
        for log_file in nodewise_log_file:
            with open(os.path.join(log_file), 'r') as f:
                preds += [(l.split('@@')[-3], l.split('@@')[-2]) for l in f.readlines()]
        scores = []
        for t in preds:
            scores.append(t[0] == t[1])
        logger.info(f'\nFinal Score (Accuracy): {round(sum(scores) / len(scores), 3)}\n\tover {len(scores)} examples.\nckpt_path: {ckpt_path}')


class OCRTrainer(Trainer):

    def __init__(self, out_dir=None, **kwargs):
        self.is_wds = kwargs.pop('is_wds', False)
        super().__init__(**kwargs)
        self.compute_metrics: BaseMetric = VQAMetrics(tokenizer=self.tokenizer)
        self.generation_config = self.model.generation_config

        anls_iou_threshold = getattr(self.args, "anls_iou_threshold", 0.0)
        self.anls_metric = ANLSMetric(iou_threshold=anls_iou_threshold)
        self.generate_max_length = self.model.language_model_tokenizer.generate_max_new_tokens_len
        self.mt_logs = {
            'ocr': 0.,
            'mpm': 0.
        }

    def _truncate_padded_input(self, inputs):
        padding_side = self.tokenizer.padding_side
        if 'attention_mask' not in inputs:
            return inputs
        non_zero = torch.nonzero(inputs["attention_mask"].sum(0), as_tuple=True)
        if len(non_zero[0] > 0):
            if padding_side == 'left':
                first_nonzero_idx = non_zero[0][0].item()
                if first_nonzero_idx % 4 != 0:
                    first_nonzero_idx = (first_nonzero_idx // 4) * 4
                for k, v in inputs.items():
                    if "image" not in k and isinstance(v[0], torch.Tensor):
                        inputs[k] = v[:, first_nonzero_idx:].contiguous()
            else:
                last_nonzero_idx = non_zero[0][-1].item() + 1
                if last_nonzero_idx % 4 != 0:
                    last_nonzero_idx = (1 + last_nonzero_idx // 4) * 4
                for k, v in inputs.items():
                    if "image" not in k and isinstance(v[0], torch.Tensor):
                        inputs[k] = v[:, :last_nonzero_idx].contiguous()
        return inputs

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.is_wds and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            node = torch.distributed.get_rank()
            size = torch.distributed.get_world_size()
            inputs = {k: v[node::size] for k, v in inputs.items()}
            inputs = self._truncate_padded_input(inputs)

        inputs = self._prepare_inputs(inputs)
        vqa_former_visual_feature = inputs['image'] if 'image' in inputs else None

        model_inputs = {k: v for k, v in inputs.items() if k in {'image', 'labels'}}
        if self.model.task_name == 'mpm_alt_concat':
            model_inputs['input_ids'] = inputs['input_ids']

        # generate 1 every 10
        is_to_generate = random.random() > 0.9

        if is_to_generate:
            node = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if  node == 0:
                outputs = getattr(model , 'module', model).generate(**model_inputs,
                                                                    do_sample=False,
                                                                    top_p=0.9,
                                                                    temperature=1,
                                                                    num_beams=5,
                                                                    max_new_tokens=64,
                                                                    min_length=1,
                                                                    repetition_penalty=1.5,
                                                                    length_penalty=1.0,
                                                                    num_return_sequences=1)
                prediction_ids = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

                predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=self.model.task_name != 'mpm_alt_concat')
                labels = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=self.model.task_name != 'mpm_alt_concat')

                # logging some answers
                str_to_print = ''
                for i in range(len(predictions)):
                    str_to_print += f"\nGT:\n\t{labels[i]}\n\npred:\n\t{predictions[i]}\n"
                logger.info(str_to_print)
            loss = None
        else:
            with torch.no_grad():
                outputs = model(**model_inputs)
            loss = outputs['loss']

            # dist.barrier()

        return (loss, None, None)

    def calculate_losses_and_closest_answer(self, inputs, predictions):
        scorer = AnlsScorer()
        preds = scorer.from_items(predictions)
        annots = scorer.from_items(inputs["answer_list"])
        scorer.add(preds, annots)
        losses = scorer.scores
        indices = scorer.indices
        closet_answers = [ans[indices[index]] for index, ans in enumerate(inputs["answer_list"])]
        return losses, closet_answers

    def _get_wds_loader(self, split, **kwargs):

        if split == 'train':
            dataset = self.train_dataset
            data_collator = self.data_collator
            batch_size = self._train_batch_size
        elif split == 'eval':
            eval_dataset = kwargs.pop("eval_dataset", None)
            dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            batch_size = self.args.eval_batch_size
        else:
            raise NotImplementedError(f"{split}")

        _kwargs = {}
        if self.args.dataloader_num_workers > 0:
            logger.warning("dataloader_num_workers > 0, set multiprocessing_context to spawn")
            _kwargs['multiprocessing_context'] = 'spawn'
        loader = wds.WebLoader(
            dataset, batch_size=None, shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            **_kwargs,
        )

        return loader

    def get_train_dataloader(self):
        if self.is_wds:
            return self._get_wds_loader(split='train')
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self.is_wds:
            return self._get_wds_loader(split='eval', eval_dataset=eval_dataset)
        else:
            return super().get_eval_dataloader()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            # if is_torch_tpu_available():
            #     xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            cumm_log_step = (self.state.global_step - self._globalstep_last_logged)

            if isinstance(self.model, VisFocus_OCR_MPM):
                ocr_loss = self._nested_gather(self.mt_logs['ocr']).mean().item()
                mpm_loss = self._nested_gather(self.mt_logs['mpm']).mean().item()
                self.mt_logs['ocr'] = 0.0
                self.mt_logs['mpm'] = 0.0
                logs['ocr_loss'] = round(ocr_loss / (cumm_log_step / 2), 4) # divide by 2 because each task steps is equally divided
                logs['mpm_loss'] = round(mpm_loss / (cumm_log_step / 2), 4) # divide by 2 because each task steps is equally divided

            logs["loss"] = round(tr_loss_scalar / cumm_log_step, 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["global_step"] = self.state.global_step
            logs["total_steps"] = self.state.max_steps


            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            if isinstance(inputs['input_ids'], dict):
                for k in ['input_ids', 'attention_mask', 'token_boxes', 'labels']:
                    inputs[k] = inputs[k][self.model.task_name]
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        if isinstance(self.model, VisFocus_OCR_MPM) and self.state.global_step % self.args.gradient_accumulation_steps == 0:
            self.mt_logs[self.model.task_name] += loss
            self.model.switch_task_name()

        return loss.detach() / self.args.gradient_accumulation_steps


    def get_parameter_names(model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result


class DocClsTrainer(Trainer):

    def __init__(self, out_dir=None, **kwargs):
        self.is_wds = kwargs.pop('is_wds', False)
        super().__init__(**kwargs)
        self.compute_metrics: BaseMetric = ClassificationMetrics(tokenizer=self.tokenizer)
        self.out_dir_metrics = os.path.join(out_dir, 'metric_logs')
        os.makedirs(self.out_dir_metrics, exist_ok=True)

        self.generation_config = self.model.generation_config

        anls_iou_threshold = getattr(self.args, "anls_iou_threshold", 0.0)
        self.anls_metric = ANLSMetric(iou_threshold=anls_iou_threshold)
        self.generate_max_length = self.model.language_model_tokenizer.generate_max_new_tokens_len

    def aggregate_metric_fn(self, ckpt_path=None):
        # accuracy for doc cls.
        nodewise_log_file = [f'{self.out_dir_metrics}/{fn}' for fn in os.listdir(self.out_dir_metrics)]
        preds = []
        for log_file in nodewise_log_file:
            with open(os.path.join(log_file), 'r') as f:
                preds += [(l.split('--')[-3], l.split('--')[-2]) for l in f.readlines()]
        scores = []
        for t in preds:
            scores.append(t[0] == t[1])
        logger.info(f'\nFinal Score (Accuracy): {round(sum(scores) / len(scores), 3)}\n\tover {len(scores)} examples.\nckpt_path: {ckpt_path}')

    def _truncate_padded_input(self, inputs):
        padding_side = self.tokenizer.padding_side
        if 'attention_mask' not in inputs:
            return inputs
        non_zero = torch.nonzero(inputs["attention_mask"].sum(0), as_tuple=True)
        if len(non_zero[0] > 0):
            if padding_side == 'left':
                first_nonzero_idx = non_zero[0][0].item()
                if first_nonzero_idx % 4 != 0:
                    first_nonzero_idx = (first_nonzero_idx // 4) * 4
                for k, v in inputs.items():
                    if "image" not in k and isinstance(v[0], torch.Tensor):
                        inputs[k] = v[:, first_nonzero_idx:].contiguous()
            else:
                last_nonzero_idx = non_zero[0][-1].item() + 1
                if last_nonzero_idx % 4 != 0:
                    last_nonzero_idx = (1 + last_nonzero_idx // 4) * 4
                for k, v in inputs.items():
                    if "image" not in k and isinstance(v[0], torch.Tensor):
                        inputs[k] = v[:, :last_nonzero_idx].contiguous()
        return inputs

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.is_wds and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            node = torch.distributed.get_rank()
            size = torch.distributed.get_world_size()
            inputs = {k: v[node::size] for k, v in inputs.items()}
            inputs = self._truncate_padded_input(inputs)

        inputs = self._prepare_inputs(inputs)
        model_inputs = {k: v for k, v in inputs.items() if k in {'image', 'labels'}}

        node = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if not model.training:
            if not hasattr(model, 'prompt_embeds'):
                # created in the first forward function (in generation it doen't use forward())
                with torch.no_grad():
                    model(**model_inputs)
            outputs = getattr(model , 'module', model).generate(**model_inputs,
                                                                do_sample=False,
                                                                top_p=0.9,
                                                                temperature=1,
                                                                num_beams=5,
                                                                max_new_tokens=64,
                                                                min_length=1,
                                                                repetition_penalty=1.5,
                                                                length_penalty=1.0,
                                                                num_return_sequences=1,
                                                                generation_config=self.generation_config)
                                                                # max_new_tokens=256) # self.generate_max_length)
            prediction_ids = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

            predictions = self.tokenizer.batch_decode(prediction_ids, skip_special_tokens=True)
            labels = self.tokenizer.batch_decode(inputs['labels'], skip_special_tokens=True)

            write_mode = getattr(self, f'write_mode_{node}', 'w')
            for i, (p, l) in enumerate(zip(predictions, labels)):
                with open(f'{self.out_dir_metrics}/error_logs_rank{node}.txt', write_mode) as f:
                    f.write(f"{i}--{p}--{l}--{int(p == l)}\n")
                write_mode = 'a' # after writing the first line, from now on append
            setattr(self, f'write_mode_{node}', 'a')

            # generate 1 every 10
            if node == 0 and random.random() > 0.9:
                # logging some answers
                str_to_print = ''
                for i in range(len(predictions)):
                    str_to_print += f"\nGT:\n\t{labels[i]}\n\npred:\n\t{predictions[i]}\n"
                logger.info(str_to_print)
            loss = None
        else:
            with torch.no_grad():
                outputs = model(**model_inputs)
            loss = outputs['loss']

        return (loss, None, None)

    def calculate_losses_and_closest_answer(self, inputs, predictions):
        scorer = AnlsScorer()
        preds = scorer.from_items(predictions)
        annots = scorer.from_items(inputs["answer_list"])
        scorer.add(preds, annots)
        losses = scorer.scores
        indices = scorer.indices
        closet_answers = [ans[indices[index]] for index, ans in enumerate(inputs["answer_list"])]
        return losses, closet_answers

    def _get_wds_loader(self, split, **kwargs):

        if split == 'train':
            dataset = self.train_dataset
            data_collator = self.data_collator
            batch_size = self._train_batch_size
        elif split == 'eval':
            eval_dataset = kwargs.pop("eval_dataset", None)
            dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            batch_size = self.args.eval_batch_size
        else:
            raise NotImplementedError(f"{split}")

        _kwargs = {}
        if self.args.dataloader_num_workers > 0:
            logger.warning("dataloader_num_workers > 0, set multiprocessing_context to spawn")
            _kwargs['multiprocessing_context'] = 'spawn'
        loader = wds.WebLoader(
            dataset, batch_size=None, shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            **_kwargs,
        )

        return loader

    def get_train_dataloader(self):
        if self.is_wds:
            return self._get_wds_loader(split='train')
        else:
            return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self.is_wds:
            return self._get_wds_loader(split='eval', eval_dataset=eval_dataset)
        else:
            return super().get_eval_dataloader()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            cumm_log_step = (self.state.global_step - self._globalstep_last_logged)

            if isinstance(self.model, VisFocus_OCR_MPM):
                ocr_loss = self._nested_gather(self.mt_logs['ocr']).mean().item()
                mpm_loss = self._nested_gather(self.mt_logs['mpm']).mean().item()
                self.mt_logs['ocr'] = 0.0
                self.mt_logs['mpm'] = 0.0
                logs['ocr_loss'] = round(ocr_loss / (cumm_log_step / 2), 4) # divide by 2 because each task steps is equally divided
                logs['mpm_loss'] = round(mpm_loss / (cumm_log_step / 2), 4) # divide by 2 because each task steps is equally divided

            logs["loss"] = round(tr_loss_scalar / cumm_log_step, 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["global_step"] = self.state.global_step
            logs["total_steps"] = self.state.max_steps


            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        if isinstance(self.model, VisFocus_OCR_MPM) and self.state.global_step % self.args.gradient_accumulation_steps == 0:
            self.mt_logs[self.model.task_name] += loss
            self.model.switch_task_name()

        return loss.detach() / self.args.gradient_accumulation_steps


    def get_parameter_names(model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result


ALL_LAYERNORM_LAYERS = [nn.LayerNorm]
