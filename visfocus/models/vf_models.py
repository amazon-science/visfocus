import torch
from torch import nn
from torch.nn import LayerNorm, CrossEntropyLoss, MSELoss, L1Loss
from torch.nn import functional as F

from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.file_utils import ModelOutput
from timm.models.layers import trunc_normal_
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import warnings
import random

if __name__ == '__main__':
    from visfocus.models.vision_model import get_vision_model, load_vision_pretrained
    from visfocus.models.base_model import BaseModel
else:
    from .vision_model import get_vision_model, load_vision_pretrained
    from .base_model import BaseModel


class VisFocusConfig(T5Config):
    def __init__(self, max_2d_position_embeddings=1024, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-12
        self.vision = None


class SpatialEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.x_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.y_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.h_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.w_position_embeddings = nn.Embedding(
            config.max_2d_position_embeddings, config.hidden_size
        )
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config

    def forward(
            self,
            bbox,
    ):
        seq_length = bbox.size(1)

        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        h_position_embeddings = self.h_position_embeddings(
            bbox[:, :, 3] - bbox[:, :, 1]
        )
        w_position_embeddings = self.w_position_embeddings(
            bbox[:, :, 2] - bbox[:, :, 0]
        )
        embeddings = (
                left_position_embeddings
                + upper_position_embeddings
                + right_position_embeddings
                + lower_position_embeddings
                + h_position_embeddings
                + w_position_embeddings
        )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class EmbedMatcher(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.embedd_matcher = nn.Sequential(
            nn.Linear(input_dim, inner_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(inner_dim, output_dim, bias=False),
            nn.Dropout(dropout_rate)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.embedd_matcher(x)
        return x


class EmbedMatcherMerger(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim, input_resolution, dropout_rate=0.1):
        from .swin_transformer_v2 import PatchMerging
        super().__init__()

        self.downsample = PatchMerging(dim=inner_dim, input_resolution=input_resolution)
        self.downsample.reduction = nn.Linear(4 * input_dim, input_dim, bias=False)
        self.downsample.norm = self.downsample.norm.__class__(input_dim)
        
        self.embedd_matcher = nn.Sequential(
            nn.Linear(input_dim, inner_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(inner_dim, output_dim, bias=False),
            nn.Dropout(dropout_rate)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downsample(x)
        x = self.embedd_matcher(x)
        return x


class EmbedMatcherConv(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim, input_resolution, dropout_rate=0.1):
        super().__init__()
        self.input_resolution = input_resolution # (H,W)
        self.output_dim = output_dim
        self.embedd_matcher = nn.Sequential(
            nn.Conv2d(input_dim, inner_dim, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(inner_dim, output_dim, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
class EmbedMatcherConv(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim, input_resolution, dropout_rate=0.1):
        super().__init__()
        self.input_resolution = input_resolution # (H,W)
        self.output_dim = output_dim
        self.embedd_matcher = nn.Sequential(
            nn.Conv2d(input_dim, inner_dim, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(inner_dim, output_dim, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )


class EmbedMatcherConv(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim, input_resolution, dropout_rate=0.1):
        super().__init__()
        self.input_resolution = input_resolution # (H,W)
        self.output_dim = output_dim
        self.embedd_matcher = nn.Sequential(
            nn.Conv2d(input_dim, inner_dim, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(inner_dim, output_dim, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B, C, H, W)
        x = self.embedd_matcher(x)
        x = x.view(B, -1, self.output_dim)
        return x


class EmbedMatcherConv_V2(nn.Module):
    def __init__(self, input_dim, inner_dim, output_dim, input_resolution, dropout_rate=0.1):
        super().__init__()
        self.input_resolution = input_resolution # (H,W)
        self.output_dim = output_dim
        self.embedd_matcher = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B, C, H, W)
        x = self.embedd_matcher(x)
        x = x.view(B, -1, self.output_dim)
        return x


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class VisFocus(T5ForConditionalGeneration, BaseModel):
    def __init__(self, config, logger):
        super().__init__(config)
        self.set_task_name('ocr')
        self.model_arch = 'visfocus'
        self.config = config
        self.logger = logger
        if config.vision is not None:
            self.vision_model = get_vision_model(config.vision)

        if self.vision_model.model_name in ['swin_v2', 'mvit_v2', 'docswin_v1', 'vit_clip', 'hivit', 'lit_v2']:
            input_dim = self.vision_model.num_features

            matcher = MATCHER_MAP[self.config.matcher_type]

            if hasattr(self.vision_model, 'last_ds'):
                input_dim = self.vision_model.last_ds.norm.normalized_shape[0]
                
            if matcher == EmbedMatcher:
                self.vision_embed_matcher = matcher(
                    input_dim,
                    config.hidden_size,
                    config.hidden_size,
                    config.hidden_dropout_prob
                )
            elif matcher == EmbedMatcherMerger or matcher == EmbedMatcherConv or matcher == EmbedMatcherConv_V2:
                self.vision_embed_matcher = matcher(
                    input_dim=input_dim,
                    inner_dim=config.hidden_size,
                    output_dim=config.hidden_size,
                    dropout_rate=config.hidden_dropout_prob,
                    # for patch-merging
                    input_resolution=self.vision_model.layers[-1].input_resolution
                )

            if config.vision.model.vision_resume_from:
                load_vision_pretrained(config.vision, self, logger)

        self.freezed_modules = getattr(config, 'freeze_modules', [])
        if self.freezed_modules:
            modules = [getattr(self, mname) for mname in config.freeze_modules]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        logger.warning(f'\nfreezed modules in {type(self).__name__}: {self.freezed_modules}')

        unfreeze_modules = getattr(config, 'unfreeze_modules', [])
        unfreeze_module_names = []
        if unfreeze_modules:
            for n, param in self.named_parameters():
                param.requires_grad = any([m in n for m in unfreeze_modules])
                if param.requires_grad:
                    unfreeze_module_names.append(n)
        logger.warning(f'\nunfreezed modules in {type(self).__name__}: {unfreeze_module_names}')

        # losses
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

        self.init_weights()
        
        ### indepedant module init:
        if getattr(config.vision.model.swinv2, 'downsampling_method', '') == 'context_merging':
            from visfocus.models.swin_transformer_v2 import ContextPatchMerging
            for l in self.vision_model.layers:
                if l.downsample is not None and isinstance(l.downsample, ContextPatchMerging):
                    l.downsample.mgq_attn._reset_parameters()

        if self.config.lora is not None:
            self.apply_lora()

        if self.config.vl_l1_loss:
            self.vl_l1_loss_fct = L1Loss()

    def load_vision_pretrained(self):
        if self.vision_model.model_name in ['swin_v2', 'docswin_v1', 'vit_clip', 'hivit', 'lit_v2'] and self.config.vision.model.vision_resume_from:
            load_vision_pretrained(self.config.vision, self, self.logger)
        elif self.vision_model.model_name == 'mvit_v2':
            self.logger.warning(f'Loaded {self.config.vision.model.variant}.{self.config.vision.model.pretrained_cfg} weight without relative positions due to resolution mismatch.')                

    def encoder_decoder_forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        r"""
        https://huggingface.co/transformers/v4.5.1/_modules/transformers/modeling_t5.html#T5ForConditionalGeneration.forward
        or https://huggingface.co/transformers/_modules/transformers/modeling_t5.html#T5ForConditionalGeneration.forward
        """

        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
            if self.config.vl_l1_loss:
                labels_ = labels.clone()
                labels_[labels_ == -100] = self.input_tokenizer.pad_token_id # -> replace the ignore_index with the pad_token id to calculate the text target for the vl loss
                with torch.no_grad():
                    target = self.encoder(input_ids=labels_).last_hidden_state
                if target.shape[1] != hidden_states.shape[1]:
                    v_encoder_intrp = F.interpolate(hidden_states.permute(0,2,1), size=target.shape[1], mode='linear').permute(0,2,1)
                    vl_loss =  (50 * self.vl_l1_loss_fct(v_encoder_intrp, target))
                    loss += vl_loss
                    if random.random() > 0.9:
                        self.logger.warning(f'vl_loss: {vl_loss.item()}')
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            if loss is not None:
                output = ((loss,) + output)

            return output

        seq2seq_output = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        return seq2seq_output

    def forward(self,
                input_ids=None,
                bbox=None,
                image=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                **kwargs):
        # see https://huggingface.co/transformers/v2.10.0/_modules/transformers/modeling_t5.html#T5Model.forward
        
        if not kwargs.get('encoder_outputs'):
            _, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=None, image=image)
        else:
            # for generation mode
            assert kwargs.get('decoder_input_ids') is not None
            _ = vision_embeds = attention_mask = None

        return self.encoder_decoder_forward(input_ids=None,
                                            attention_mask=attention_mask,
                                            encoder_outputs=kwargs.get('encoder_outputs'),
                                            decoder_input_ids=kwargs.get('decoder_input_ids'),
                                            decoder_attention_mask=None,
                                            head_mask=head_mask,
                                            decoder_head_mask=None,
                                            past_key_values=kwargs.get('past_key_values'),
                                            inputs_embeds=vision_embeds,
                                            decoder_inputs_embeds=kwargs.get('decoder_inputs_embeds'),
                                            labels=labels,
                                            use_cache=True,
                                            output_attentions=kwargs.get('output_attentions'),
                                            output_hidden_states=kwargs.get('output_hidden_states'),
                                            return_dict=kwargs.get('return_dict')
                                            )


    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        if kwargs.get('encoder_outputs') is not None:
            return {'attention_mask': kwargs.get('attention_mask'),
                    'encoder_outputs': kwargs.get('encoder_outputs'),
                    'decoder_input_ids': input_ids,
                    'past_key_values': kwargs.get('past'),
                    }
        else:
            raise ValueError(
                "Make sure that encoder_outputs is already computed when preapring inputs for generation. --y.x.")

    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        # text embedding
        batch_size = image.shape[0]

        if input_ids is not None:
            text_embeds = self.shared(input_ids)
            text_seq_length = text_embeds.shape[1]
        else:
            text_embeds = None
            text_seq_length = 0
                
        assert self.config.vision is not None
        # vision embedding
        vision_embeds = self.vision_model(image)
        if self.vision_model.model_name in ["swin_v2", "mvit_v2", 'docswin_v1', 'vit_clip', 'hivit', 'lit_v2']:
            vision_embeds = self.vision_embed_matcher(vision_embeds)
        elif self.vision_model.model_name == "mlp_dfv2":
            vision_embeds = vision_embeds[0] + vision_embeds[1]
        vision_seq_length = vision_embeds.shape[1]
        # add task token (e.g <OCR> for ocr)
        vision_embeds, text_seq_length = self.concat_task_token(vision_embeds, text_seq_length)
        attention_mask = torch.ones((batch_size, vision_seq_length + text_seq_length), dtype=torch.int32).to(self.device)
        return text_embeds, vision_embeds, attention_mask

    def concat_task_token(self, embeds, text_seq_length=0):
        # add task token (e.g <OCR> for ocr)
        if self.task_name in self.task_token_ids.keys():
            B = embeds.shape[0]
            task_embeds = self.shared(self.task_token_ids[self.task_name])
            text_seq_length += task_embeds.shape[0]
            return torch.cat((embeds, task_embeds.repeat((B, 1, 1))), dim=1), text_seq_length
        else:
            # no such task token exists
            return embeds, text_seq_length

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        input_name = 'inputs_embeds'
        _, vision_embeds, attention_mask = self._prepare_encoder_inputs(image=model_kwargs['image'])
        model_kwargs['attention_mask'] = attention_mask

        inputs = vision_embeds

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        assert "encoder_outputs" not in model_kwargs

        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        irrelevent_fields = ['input_ids', 'attention_mask', 'inputs_embeds', 'image', 'bbox', 'line_coordinates',
                             'adj', 'lm_labels', 'banned_token_ids', 'questions', 'answers', 'labels', 'task_name']
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix) and argument not in irrelevent_fields
        }

        # 3. make sure that encoder returns `ModelOutput`
        encoder_kwargs["return_dict"] = True
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(
            input_ids=None, attention_mask=model_kwargs['attention_mask'],
            inputs_embeds=inputs_tensor, **encoder_kwargs)

        return model_kwargs

    def add_task_tokens(self):
        self.input_tokenizer.add_tokens('<OCR>', special_tokens=True)
        self.task_token_ids = torch.nn.ParameterDict([['ocr', self.register_token('<OCR>')]])

    def register_token(self, token: str):
        self.input_tokenizer.add_tokens(token, special_tokens=True)
        token_ids = self.input_tokenizer.encode(token)
        return torch.nn.Parameter(torch.tensor(token_ids), requires_grad=False)
    
    def set_task_name(self, task_name):
        if task_name:
            self.task_name = task_name

    def apply_lora(self):
        from peft import LoraModel, LoraConfig
        lora_config = LoraConfig(**self.config.lora)
        self.encoder = LoraModel(self.encoder, lora_config, 'default').model
        self.decoder  = LoraModel(self.decoder , lora_config, 'default').model
        self.logger.warning(f'LoRa is applied in LM on {self.config.lora.target_modules}')
        _lm_total = sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.decoder.parameters())
        _lm_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) + sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        self.logger.warning(f'LoRa uses {_lm_trainable} / {_lm_total} trainable params')

    def get_trivial_mask(self, inp):
        return torch.ones((inp.shape[:2]), dtype=torch.int32).to(self.device)


class VisFocus_MPM(VisFocus):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('mpm')
        if hasattr(config.vision.model.swinv2, 'text_embedder'):
            if config.vision.model.swinv2.text_embedder == 'shared':
                self.text_embedder = self.encoder
            elif config.vision.model.swinv2.text_embedder == 'tiny_bert':
                from visfocus.models.bert_encoder import BERT
                self.text_embedder = BERT(variant='tiny')
            else:
                from visfocus.models.t5_encoder import T5_Encoder
                self.text_embedder = T5_Encoder(config.vision.model.swinv2.text_embedder, freeze=True)
        else:
            self.text_embedder = self.shared
        
    def forward(self,
                input_ids=None,
                bbox=None,
                image=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                **kwargs):        
        if not kwargs.get('encoder_outputs'):
            if self.task_name == 'ocr':
                input_ids = None
                if not hasattr(self, 'prompt_embeds'):
                    prompt = 'what is written in this document?'
                    prompt_ids = self.input_tokenizer.encode(prompt)
                    B = image.shape[0]
                    prompt_ids = torch.tensor(prompt_ids).expand(B, len(prompt_ids)).to(self.device)
                    setattr(self, 'prompt_embeds', self.text_embedder(prompt_ids).detach())
            _, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=input_ids, image=image)
        else:
            # for generation mode
            assert kwargs.get('decoder_input_ids') is not None
            _ = vision_embeds = attention_mask = None

        return self.encoder_decoder_forward(input_ids=None,
                                            attention_mask=attention_mask,
                                            encoder_outputs=kwargs.get('encoder_outputs'),
                                            decoder_input_ids=kwargs.get('decoder_input_ids'),
                                            decoder_attention_mask=None,
                                            head_mask=head_mask,
                                            decoder_head_mask=None,
                                            past_key_values=kwargs.get('past_key_values'),
                                            inputs_embeds=vision_embeds,
                                            decoder_inputs_embeds=kwargs.get('decoder_inputs_embeds'),
                                            labels=labels,
                                            use_cache=True,
                                            output_attentions=kwargs.get('output_attentions'),
                                            output_hidden_states=kwargs.get('output_hidden_states'),
                                            return_dict=kwargs.get('return_dict')
                                            )
    
    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        batch_size = image.shape[0]

        # if prompt is contant
        if self.task_name == 'ocr':
            assert input_ids is None
            text_embeds = self.prompt_embeds
        else:
            assert input_ids is not None
            try:
                if self.text_embedder == self.encoder:
                    with torch.no_grad():
                        text_embeds = self.encoder(input_ids).last_hidden_state
                else:
                    text_embeds = self.text_embedder(input_ids)

                text_embeds = text_embeds.detach()
            except:
                # for debug
                print('------------->>>>>>>>>>>>>>>>>>', input_ids.dtype)
                print(input_ids)
                text_embeds = self.text_embedder(input_ids.int()).detach()

        text_seq_length = text_embeds.shape[1] if self.task_name == 'pm_vqa_concat' else 0
        assert self.config.vision is not None
        # vision embedding
        vision_embeds = self.vision_model(image, context_prompts=text_embeds)
        if self.vision_model.model_name in ["swin_v2"]:
            vision_embeds = self.vision_embed_matcher(vision_embeds)
        vision_seq_length = vision_embeds.shape[1]
        # add task token (e.g <OCR> for ocr)
        vision_embeds, text_seq_length = self.concat_task_token(vision_embeds, text_seq_length=text_seq_length)
        attention_mask = torch.ones((batch_size, vision_seq_length + text_seq_length), dtype=torch.int32).to(self.device)
        return text_embeds, vision_embeds, attention_mask

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """

        input_name = 'inputs_embeds'
        _, vision_embeds, attention_mask = self._prepare_encoder_inputs(image=model_kwargs['image'], input_ids=model_kwargs['input_ids'])
        model_kwargs['attention_mask'] = attention_mask
        inputs = vision_embeds
        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<MPM>', special_tokens=True)
        self.task_token_ids.update({'mpm': self.register_token('<MPM>')})


class VisFocus_VQAConcat(VisFocus):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        # self.set_task_token('vqa_concat')
        self.vqa_method = config.vqa_method

    def forward(self,
                questions=None,
                answers=None,
                image=None,
                labels=None,
                **kwargs):        
        if kwargs.get('encoder_outputs') is None:
            text_embeds, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=questions['input_ids'], image=image)
            inputs_embeds = torch.concat((vision_embeds, text_embeds), dim=1)
        else:
            # for generation mode (image encoding happens before)
            assert kwargs.get('decoder_input_ids') is not None
            assert kwargs.get('encoder_outputs') is not None
            inputs_embeds = kwargs.get('encoder_outputs')
            text_embeds = vision_embeds = attention_mask = None

        return self.encoder_decoder_forward(input_ids=None,
                                            attention_mask=attention_mask,
                                            encoder_outputs=kwargs.get('encoder_outputs'),
                                            decoder_input_ids=kwargs.get('decoder_input_ids'),
                                            decoder_attention_mask=None,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            past_key_values=kwargs.get('past_key_values'),
                                            inputs_embeds=inputs_embeds,
                                            decoder_inputs_embeds=kwargs.get('decoder_inputs_embeds'),
                                            labels=labels,
                                            use_cache=True,
                                            output_attentions=kwargs.get('output_attentions'),
                                            output_hidden_states=kwargs.get('output_hidden_states'),
                                            return_dict=kwargs.get('return_dict')
                                            )
    
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        input_name = 'inputs_embeds'
        text_embeds, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=model_kwargs['questions']['input_ids'], image=model_kwargs['image'])
        model_kwargs['attention_mask'] = attention_mask
        inputs_embeds = torch.concat((vision_embeds, text_embeds), dim=1)
        inputs = inputs_embeds
        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<VQAC>', special_tokens=True)
        self.task_token_ids.update({'vqa_concat': self.register_token('<VQAC>')})


class VisFocus_VQAConcat_LV(VisFocus):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.vqa_method = config.vqa_method

    def forward(self,
                questions=None,
                answers=None,
                image=None,
                labels=None,
                **kwargs):        
        if kwargs.get('encoder_outputs') is None:
            text_embeds, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=questions['input_ids'], image=image)
            inputs_embeds = torch.concat((text_embeds, vision_embeds), dim=1)
            attention_mask = self.get_trivial_mask(inputs_embeds) # -> when different tokenizer is used for ViLMA/concat, need to re-calculate attn. mask
        else:
            # for generation mode (image encoding happens before)
            assert kwargs.get('decoder_input_ids') is not None
            assert kwargs.get('encoder_outputs') is not None
            inputs_embeds = kwargs.get('encoder_outputs')
            text_embeds = vision_embeds = attention_mask = None

        return self.encoder_decoder_forward(input_ids=None,
                                            attention_mask=attention_mask,
                                            encoder_outputs=kwargs.get('encoder_outputs'),
                                            decoder_input_ids=kwargs.get('decoder_input_ids'),
                                            decoder_attention_mask=None,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            past_key_values=kwargs.get('past_key_values'),
                                            inputs_embeds=inputs_embeds,
                                            decoder_inputs_embeds=kwargs.get('decoder_inputs_embeds'),
                                            labels=labels,
                                            use_cache=True,
                                            output_attentions=kwargs.get('output_attentions'),
                                            output_hidden_states=kwargs.get('output_hidden_states'),
                                            return_dict=kwargs.get('return_dict')
                                            )
    
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        input_name = 'inputs_embeds'
        text_embeds, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=model_kwargs['questions']['input_ids'], image=model_kwargs['image'])
        model_kwargs['attention_mask'] = attention_mask
        inputs_embeds = torch.concat((text_embeds, vision_embeds), dim=1)
        inputs = inputs_embeds
        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<VQAC>', special_tokens=True)
        self.task_token_ids.update({'vqa_concat': self.register_token('<VQAC>')})


class VisFocus_MPM_VQA_CONCAT(VisFocus_MPM, VisFocus_VQAConcat):
    def forward(self, questions=None, answers=None, image=None, labels=None, **kwargs):     
        self.set_task_name('pm_vqa_concat')
        return VisFocus_VQAConcat.forward(self, questions, answers, image, labels, **kwargs)
    
    def _prepare_model_inputs(self, inputs=None, bos_token_id=None, model_kwargs=None ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        return VisFocus_VQAConcat._prepare_model_inputs(self, inputs, bos_token_id, model_kwargs)
    
    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        self.set_task_name('pm_vqa_concat')
        text_embeds, vision_embeds, attention_mask = VisFocus_MPM._prepare_encoder_inputs(self, image, input_ids, bbox, attention_mask)
        text_embeds = self.shared(input_ids) # for concat, use direct the T5 nn.embeddings
        return text_embeds, vision_embeds, attention_mask

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<MPM_VQA_CONCAT>', special_tokens=True)
        self.task_token_ids.update({'pm_vqa_concat': self.register_token('<MPM_VQA_CONCAT>')})


class VisFocus_MPM_VQA_CONCAT_FT(VisFocus_MPM):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('pm_vqa_concat_ft')
        self.q_prompt = torch.tensor([[746, 10, 3]]) # -> "question: "
        self.c_prompt = torch.tensor([[2625, 10, 3]]) # -> "context: "

    def forward(self, questions=None, answers=None, image=None, labels=None, **kwargs):     
        if kwargs.get('encoder_outputs') is None:
            text_embeds, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=questions['input_ids'], image=image)
            inputs_embeds = torch.cat([text_embeds, vision_embeds], dim=1) # -> concat prompt to vision embeds. structure: "questions: <THE QUESTION> context: vision_embeds>"
        else:
            # for generation mode (image encoding happens before)
            assert kwargs.get('decoder_input_ids') is not None
            assert kwargs.get('encoder_outputs') is not None
            inputs_embeds = kwargs.get('encoder_outputs')
            text_embeds = vision_embeds = attention_mask = None

        return self.encoder_decoder_forward(input_ids=None,
                                            attention_mask=attention_mask,
                                            encoder_outputs=kwargs.get('encoder_outputs'),
                                            decoder_input_ids=kwargs.get('decoder_input_ids'),
                                            decoder_attention_mask=None,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            past_key_values=kwargs.get('past_key_values'),
                                            inputs_embeds=inputs_embeds,
                                            decoder_inputs_embeds=kwargs.get('decoder_inputs_embeds'),
                                            labels=labels,
                                            use_cache=True,
                                            output_attentions=kwargs.get('output_attentions'),
                                            output_hidden_states=kwargs.get('output_hidden_states'),
                                            return_dict=kwargs.get('return_dict')
                                            )
    
    
    def _prepare_model_inputs(self, inputs=None, bos_token_id=None, model_kwargs=None ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        return VisFocus_VQAConcat._prepare_model_inputs(self, inputs, bos_token_id, model_kwargs)
    
    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        assert input_ids is not None
        assert self.config.vision is not None
        batch_size = image.shape[0]
        text_embeds = self.text_embedder(input_ids).detach() # -> for the PA-CA
        # vision embedding
        vision_embeds = self.vision_model(image, context_prompts=text_embeds)
        if self.vision_model.model_name in ["swin_v2"]:
            vision_embeds = self.vision_embed_matcher(vision_embeds)

        q_prompt = self.q_prompt.expand((batch_size, -1)).to(self.device)
        c_prompt = self.c_prompt.expand((batch_size, -1)).to(self.device)
        textual_ids = torch.cat([q_prompt, input_ids, c_prompt], dim=1)
        text_embeds = self.encoder.embed_tokens(textual_ids)

        text_seq_length = text_embeds.shape[1]
        vision_seq_length = vision_embeds.shape[1]

        attention_mask = torch.ones((batch_size, vision_seq_length + text_seq_length), dtype=torch.int32).to(self.device)
        return text_embeds, vision_embeds, attention_mask

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<MPM_VQA_CONCAT>', special_tokens=True)
        self.task_token_ids.update({'pm_vqa_concat': self.register_token('<MPM_VQA_CONCAT>')})


class VisFocus_MPM_VQA_CONCAT_LV(VisFocus_MPM, VisFocus_VQAConcat_LV):
    def forward(self, questions=None, answers=None, image=None, labels=None, **kwargs):     
        self.set_task_name('pm_vqa_concat')
        return VisFocus_VQAConcat_LV.forward(self, questions, answers, image, labels, **kwargs)
    
    def _prepare_model_inputs(self, inputs=None, bos_token_id=None, model_kwargs=None ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        inputs, input_name, model_kwargs =  VisFocus_VQAConcat_LV._prepare_model_inputs(self, inputs, bos_token_id, model_kwargs)
        model_kwargs['attention_mask'] = self.get_trivial_mask(inputs)
        return inputs, input_name, model_kwargs

    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        self.set_task_name('pm_vqa_concat')
        text_embeds, vision_embeds, attention_mask = VisFocus_MPM._prepare_encoder_inputs(self, image, input_ids, bbox, attention_mask)
        text_embeds = self.shared(input_ids) # for concat, use direct the T5 nn.embeddings
        return text_embeds, vision_embeds, attention_mask

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<MPM_VQA_CONCAT>', special_tokens=True)
        self.task_token_ids.update({'pm_vqa_concat': self.register_token('<MPM_VQA_CONCAT>')})


class VisFocus_MPM_VQA(VisFocus_MPM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.set_task_name('pm_vqa')

    def forward(self, questions=None, answers=None, image=None, labels=None, task_name='pm_vqa', **kwargs):
        if questions:
            questions = questions['input_ids']
        return VisFocus_MPM.forward(self, input_ids=questions, image=image, labels=labels, **kwargs)

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """

        input_name = 'inputs_embeds'
        _, vision_embeds, attention_mask = self._prepare_encoder_inputs(image=model_kwargs['image'], input_ids=model_kwargs['questions']['input_ids'])
        model_kwargs['attention_mask'] = attention_mask
        inputs = vision_embeds
        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<MPM_VQA>', special_tokens=True)
        self.task_token_ids.update({'pm_vqa': self.register_token('<MPM_VQA>')})


class VisFocus_MPM_DocCls(VisFocus_MPM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model_arch = 'ocrf_for_doc_cls'
        self.set_task_name('doc_cls')

    def forward(self, image=None, labels=None, **kwargs):
        if not hasattr(self, 'prompt_embeds'):
            prompt = 'What is the class of this document?'
            prompt_ids = self.input_tokenizer.encode(prompt)
            B = image.shape[0]
            prompt_ids = torch.tensor(prompt_ids).expand(B, len(prompt_ids)).to(self.device)
            setattr(self, 'prompt_embeds', self.text_embedder(prompt_ids).detach())
        return VisFocus_MPM.forward(self, input_ids=None, image=image, labels=labels, **kwargs)

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """

        input_name = 'inputs_embeds'
        _, vision_embeds, attention_mask = self._prepare_encoder_inputs(image=model_kwargs['image'])
        model_kwargs['attention_mask'] = attention_mask
        inputs = vision_embeds
        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        batch_size = image.shape[0]

        assert self.config.vision is not None
        assert input_ids is None

        text_embeds = self.prompt_embeds
        text_seq_length = 0 # no concatenation    
        # vision embedding
        vision_embeds = self.vision_model(image, context_prompts=text_embeds)
        if self.vision_model.model_name in ["swin_v2"]:
            vision_embeds = self.vision_embed_matcher(vision_embeds)
        vision_seq_length = vision_embeds.shape[1]
        attention_mask = torch.ones((batch_size, vision_seq_length + text_seq_length), dtype=torch.int32).to(self.device)
        return text_embeds, vision_embeds, attention_mask

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<DOC_CLS>', special_tokens=True)
        self.task_token_ids.update({'doc_cls': self.register_token('<DOC_CLS>')})

    def add_cls_tokens(self, dataset):
        class_tokens = DOC_CLS_MAP[dataset]
        self.classes = class_tokens
        self.input_tokenizer.add_tokens(class_tokens, special_tokens=False)


class VisFocus_OCR_MPM(VisFocus_MPM):
    def forward(self, image=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if not kwargs.get('encoder_outputs'):
            input_ids = input_ids[self.task_name]
            attention_mask = attention_mask[self.task_name]
            labels = labels[self.task_name]
        else:
            # for generation mode
            assert kwargs.get('decoder_input_ids') is not None
            attention_mask = None

        return VisFocus_MPM.forward(self, image=image, input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def _prepare_encoder_inputs(self, **kwargs):
        text_embeds, vision_embeds, _ = VisFocus_MPM._prepare_encoder_inputs(self, **kwargs)

        vision_embeds, _ = self.concat_task_token(vision_embeds)

        attention_mask = torch.ones((vision_embeds.shape[0], vision_embeds.shape[1]), dtype=torch.int32).to(self.device)
        return text_embeds, vision_embeds, attention_mask

    def switch_task_name(self):
        if self.task_name == 'ocr':
            self.set_task_name('mpm')
        elif self.task_name == 'mpm':
            self.set_task_name('ocr')
        else:
            raise AttributeError('Invalid task name (can be either ocr or mpm)')


class VisFocus_MPM_ALTERNATE_CONCAT(VisFocus_MPM):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('mpm_alt_concat')
    
    def _prepare_encoder_inputs(self, **kwargs):
        _, vision_embeds, attention_mask = VisFocus_MPM._prepare_encoder_inputs(self, **kwargs)
        
        if random.random() > 0.5 or not self.training: # alternate weather concat the questions to the image feature as the LM input
            text_embeds = self.shared(kwargs['input_ids'])
            vision_embeds = torch.concat((text_embeds, vision_embeds), dim=1) # (not ''vision" emmbeds anymore..)
            attention_mask = self.get_trivial_mask(vision_embeds)
        return _, vision_embeds, attention_mask

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<mpm_alt_concat>', special_tokens=True)
        self.task_token_ids.update({'mpm_alt_concat': self.register_token('<MPM_ALT_CONCAT>')})


class VisFocus_MPM_ALTERNATE_CONCAT_LV(VisFocus_MPM):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('mpm_alt_concat')
    
    def _prepare_encoder_inputs(self, **kwargs):
        _, vision_embeds, attention_mask = VisFocus_MPM._prepare_encoder_inputs(self, **kwargs)
        
        if random.random() > 0.5 or not self.training: # alternate weather concat the questions to the image feature as the LM input
            text_embeds = self.shared(kwargs['input_ids'])
            vision_embeds = torch.concat((text_embeds, vision_embeds), dim=1) # (not ''vision" emmbeds anymore..)
            attention_mask = self.get_trivial_mask(vision_embeds)
        return _, vision_embeds, attention_mask

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        input_name = 'inputs_embeds'
        text_embeds, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=model_kwargs['questions']['input_ids'], image=model_kwargs['image'])
        model_kwargs['attention_mask'] = attention_mask
        inputs_embeds = torch.concat((text_embeds, vision_embeds), dim=1)
        inputs = inputs_embeds
        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<mpm_alt_concat>', special_tokens=True)
        self.task_token_ids.update({'mpm_alt_concat': self.register_token('<MPM_ALT_CONCAT>')})


class VisFocus_MPM_CONCAT_LV(VisFocus_MPM):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('mpm_alt_concat_lv')
    
    def _prepare_encoder_inputs(self, **kwargs):
        _, vision_embeds, attention_mask = VisFocus_MPM._prepare_encoder_inputs(self, **kwargs)
        
        text_embeds = self.shared(kwargs['input_ids'])
        vision_embeds = torch.concat((text_embeds, vision_embeds), dim=1) # (not ''vision" emmbeds anymore..)
        attention_mask = self.get_trivial_mask(vision_embeds)
        return _, vision_embeds, attention_mask

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<mpm_alt_concat_lv>', special_tokens=True)
        self.task_token_ids.update({'mpm_alt_concat_lv': self.register_token('<MPM_ALT_CONCAT_LV>')})


class VisFocus_MPM_CONCAT_ONLY_LV(VisFocus_MPM):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('mpm_concat_only_lv')
    
    def _prepare_encoder_inputs(self, **kwargs):
        _, vision_embeds, attention_mask = VisFocus._prepare_encoder_inputs(self, **kwargs)
        
        text_embeds = self.shared(kwargs['input_ids'])
        vision_embeds = torch.concat((text_embeds, vision_embeds), dim=1) # (not ''vision" emmbeds anymore..)
        attention_mask = self.get_trivial_mask(vision_embeds)
        return _, vision_embeds, attention_mask

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<mpm_concat_only_lv>', special_tokens=True)
        self.task_token_ids.update({'mpm_concat_only_lv': self.register_token('<MPM_CONCAT_ONLY_LV>')})


class VisFocus_REORDER(VisFocus_MPM):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('reorder')

    def forward(self,
                input_ids=None,
                bbox=None,
                image=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                **kwargs):        
        if not kwargs.get('encoder_outputs'):
            _, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=input_ids, image=image)
        else:
            # for generation mode
            assert kwargs.get('decoder_input_ids') is not None
            _ = vision_embeds = attention_mask = None

        return self.encoder_decoder_forward(input_ids=None,
                                            attention_mask=attention_mask,
                                            encoder_outputs=kwargs.get('encoder_outputs'),
                                            decoder_input_ids=kwargs.get('decoder_input_ids'),
                                            decoder_attention_mask=None,
                                            head_mask=head_mask,
                                            decoder_head_mask=None,
                                            past_key_values=kwargs.get('past_key_values'),
                                            inputs_embeds=vision_embeds,
                                            decoder_inputs_embeds=kwargs.get('decoder_inputs_embeds'),
                                            labels=labels,
                                            use_cache=True,
                                            output_attentions=kwargs.get('output_attentions'),
                                            output_hidden_states=kwargs.get('output_hidden_states'),
                                            return_dict=kwargs.get('return_dict')
                                            )

    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        batch_size = image.shape[0]
        assert input_ids is not None

        text_embeds = self.text_embedder(input_ids).detach()
        perm_indices = torch.randperm(text_embeds.shape[1], device=self.device)
        text_embeds = text_embeds.index_select(1, perm_indices)
        text_seq_length = text_embeds.shape[1] if self.task_name == 'pm_vqa_concat' else 0
        # vision embedding
        assert self.config.vision is not None
        vision_embeds = self.vision_model(image, context_prompts=text_embeds)
        if self.vision_model.model_name in ["swin_v2"]:
            vision_embeds = self.vision_embed_matcher(vision_embeds)
        vision_seq_length = vision_embeds.shape[1]
        # add task token (e.g <OCR> for ocr)
        vision_embeds, text_seq_length = self.concat_task_token(vision_embeds, text_seq_length=text_seq_length)

        attention_mask = torch.ones((batch_size, vision_seq_length + text_seq_length), dtype=torch.int32).to(self.device)
        return text_embeds, vision_embeds, attention_mask

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<REORDER>', special_tokens=True)
        self.task_token_ids.update({'reorder': self.register_token('<REORDER>')})


class VisFocus_NSP(VisFocus_MPM):
    """
     NSP: Next Sentence Prediction.
    """
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('nsp')

    def forward(self, input_ids=None,
                bbox=None,
                image=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=None,
                **kwargs):
        
        if not kwargs.get('encoder_outputs'):
            _, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=input_ids, image=image)
        else:
            # for generation mode
            assert kwargs.get('decoder_input_ids') is not None
            _ = vision_embeds = attention_mask = None

        return self.encoder_decoder_forward(input_ids=None,
                                            attention_mask=attention_mask,
                                            encoder_outputs=kwargs.get('encoder_outputs'),
                                            decoder_input_ids=kwargs.get('decoder_input_ids'),
                                            decoder_attention_mask=None,
                                            head_mask=head_mask,
                                            decoder_head_mask=None,
                                            past_key_values=kwargs.get('past_key_values'),
                                            inputs_embeds=vision_embeds,
                                            decoder_inputs_embeds=kwargs.get('decoder_inputs_embeds'),
                                            labels=labels,
                                            use_cache=True,
                                            output_attentions=kwargs.get('output_attentions'),
                                            output_hidden_states=kwargs.get('output_hidden_states'),
                                            return_dict=kwargs.get('return_dict')
                                            )

    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        batch_size = image.shape[0]
        assert input_ids is not None

        for i in range(batch_size):
            hl = text_embeds[i].shape[0] // 2
            text_embeds[i, :hl]
        
        text_embeds = self.text_embedder(input_ids).detach()
        perm_indices = torch.randperm(text_embeds.shape[1], device=self.device)
        text_embeds = text_embeds.index_select(1, perm_indices)
        text_seq_length = text_embeds.shape[1] if self.task_name == 'pm_vqa_concat' else 0
        # vision embedding
        assert self.config.vision is not None
        vision_embeds = self.vision_model(image, context_prompts=text_embeds)
        if self.vision_model.model_name in ["swin_v2"]:
            vision_embeds = self.vision_embed_matcher(vision_embeds)
        vision_seq_length = vision_embeds.shape[1]
        # add task token (e.g <OCR> for ocr)
        vision_embeds, text_seq_length = self.concat_task_token(vision_embeds, text_seq_length=text_seq_length)
        attention_mask = torch.ones((batch_size, vision_seq_length + text_seq_length), dtype=torch.int32).to(self.device)
        return text_embeds, vision_embeds, attention_mask

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<NSP>', special_tokens=True)
        self.task_token_ids.update({'nsp': self.register_token('<NSP>')})


class VisFocus_MPM_MULTIRES(VisFocus_MPM_CONCAT_LV):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('mpm_mr')

    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        batch_size = image.shape[0]
        assert input_ids is not None
        if self.text_embedder == self.encoder:
            with torch.no_grad():
                text_embeds = self.encoder(input_ids).last_hidden_state
        else:
            text_embeds = self.text_embedder(input_ids)
        text_embeds = text_embeds.detach()
        text_seq_length = text_embeds.shape[1] if self.task_name == 'pm_vqa_concat' else 0    
        assert self.config.vision is not None
        # vision embedding
        v1 = rearrange(image, 'b c (h x1) (w x2) -> (b h w) c x1 x2', x1=256, x2=256)
        vision_embeds = self.vision_model(v1, context_prompts=text_embeds.repeat(18, 1, 1))
        vision_embeds = rearrange(vision_embeds, '(b h w) c d -> b (c h w) d', h=1536 // 256, w=768 // 256)
        if self.vision_model.model_name in ["swin_v2"]:
            vision_embeds = self.vision_embed_matcher(vision_embeds)
        vision_seq_length = vision_embeds.shape[1]
        # add task token (e.g <OCR> for ocr)
        vision_embeds, text_seq_length = self.concat_task_token(vision_embeds, text_seq_length=text_seq_length)
        attention_mask = torch.ones((batch_size, vision_seq_length + text_seq_length), dtype=torch.int32).to(self.device)
        return text_embeds, vision_embeds, attention_mask
    
    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<MPM_MR>', special_tokens=True)
        self.task_token_ids.update({'mpm_mr': self.register_token('<MPM_MR>')})


class VisFocus_MPM_VQA_CONCAT_LV_MR(VisFocus_MPM_MULTIRES, VisFocus_VQAConcat_LV):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.set_task_name('pm_vqa_concat_mr')

    def forward(self,
                questions=None,
                answers=None,
                image=None,
                labels=None,
                **kwargs):        
        if kwargs.get('encoder_outputs') is None:
            text_embeds, vision_embeds, _ = self._prepare_encoder_inputs(input_ids=questions['input_ids'], image=image)
            inputs_embeds = torch.concat((text_embeds, vision_embeds), dim=1)
            attention_mask = self.get_trivial_mask(inputs_embeds)
        else:
            # for generation mode (image encoding happens before)
            assert kwargs.get('decoder_input_ids') is not None
            assert kwargs.get('encoder_outputs') is not None
            inputs_embeds = kwargs.get('encoder_outputs')
            text_embeds = vision_embeds = attention_mask = None

        return self.encoder_decoder_forward(input_ids=None,
                                            attention_mask=attention_mask,
                                            encoder_outputs=kwargs.get('encoder_outputs'),
                                            decoder_input_ids=kwargs.get('decoder_input_ids'),
                                            decoder_attention_mask=None,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            past_key_values=kwargs.get('past_key_values'),
                                            inputs_embeds=inputs_embeds,
                                            decoder_inputs_embeds=kwargs.get('decoder_inputs_embeds'),
                                            labels=labels,
                                            use_cache=True,
                                            output_attentions=kwargs.get('output_attentions'),
                                            output_hidden_states=kwargs.get('output_hidden_states'),
                                            return_dict=kwargs.get('return_dict')
                                            )

    
    def _prepare_model_inputs(self, inputs=None, bos_token_id=None, model_kwargs=None ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        return VisFocus_VQAConcat_LV._prepare_model_inputs(self, inputs, bos_token_id, model_kwargs)
    
    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        self.set_task_name('pm_vqa_concat')
        text_embeds, vision_embeds, attention_mask = VisFocus_MPM_MULTIRES._prepare_encoder_inputs(self, image, input_ids, bbox, attention_mask)
        text_embeds = self.shared(input_ids) # for concat, use direct the T5 nn.embeddings
        return text_embeds, vision_embeds, attention_mask


    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<MPM_VQA_CONCAT_MR>', special_tokens=True)
        self.task_token_ids.update({'pm_vqa_concat_mr': self.register_token('<MPM_VQA_CONCAT_MR>')})


class VisFocus_MPM_VQA_CONCAT_LV_POOL(VisFocus_MPM_VQA_CONCAT_LV):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.m = nn.MaxPool2d(2, stride=2)

    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        batch_size = image.shape[0]

        # if prompt is contant
        if self.task_name == 'ocr':
            assert input_ids is None
            text_embeds = self.prompt_embeds
        else:
            assert input_ids is not None
            if self.text_embedder == self.encoder:
                with torch.no_grad():
                    text_embeds = self.encoder(input_ids).last_hidden_state
            else:
                text_embeds = self.text_embedder(input_ids)
            text_embeds = text_embeds.detach()

        text_seq_length = text_embeds.shape[1] if self.task_name == 'pm_vqa_concat' else 0
        assert self.config.vision is not None
        # vision embedding
        vision_embeds = self.vision_model(image, context_prompts=text_embeds)
        ######
        x = rearrange(vision_embeds, 'b (h w) d -> b h w d', h=384 // 8, w=192 // 8)
        x = x.permute(0, 3, 1, 2)
        x = self.m(x)
        x = x.permute(0, 2, 3, 1)
        vision_embeds = x.flatten(1, 2)
        ######

        if self.vision_model.model_name in ["swin_v2"]:
            vision_embeds = self.vision_embed_matcher(vision_embeds)
        vision_seq_length = vision_embeds.shape[1]
        # add task token (e.g <OCR> for ocr)
        vision_embeds, text_seq_length = self.concat_task_token(vision_embeds, text_seq_length=text_seq_length)
        attention_mask = torch.ones((batch_size, vision_seq_length + text_seq_length), dtype=torch.int32).to(self.device)
        text_embeds = self.shared(input_ids)
        # vision_embeds = torch.concat((text_embeds, vision_embeds), dim=1) # (not ''vision" emmbeds anymore..)
        # attention_mask = self.get_trivial_mask(vision_embeds)
        return text_embeds, vision_embeds, attention_mask


class VisFocus_VQA_P2S(VisFocus):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        # self.set_task_token('vqa_p2s')
        self.vqa_method = config.vqa_method

    def forward(self,
                questions=None,
                answers=None,
                image=None,
                labels=None,
                **kwargs):        
        if kwargs.get('encoder_outputs') is None:
            _, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=questions['input_ids'], image=image)
            inputs_embeds = vision_embeds
        else:
            # for generation mode (image encoding happens before)
            assert kwargs.get('decoder_input_ids') is not None
            assert kwargs.get('encoder_outputs') is not None
            inputs_embeds = kwargs.get('encoder_outputs')
            text_embeds = vision_embeds = attention_mask = None

        return self.encoder_decoder_forward(input_ids=None,
                                            attention_mask=attention_mask,
                                            encoder_outputs=kwargs.get('encoder_outputs'),
                                            decoder_input_ids=kwargs.get('decoder_input_ids'),
                                            decoder_attention_mask=None,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            past_key_values=kwargs.get('past_key_values'),
                                            inputs_embeds=inputs_embeds,
                                            decoder_inputs_embeds=kwargs.get('decoder_inputs_embeds'),
                                            labels=labels,
                                            use_cache=True,
                                            output_attentions=kwargs.get('output_attentions'),
                                            output_hidden_states=kwargs.get('output_hidden_states'),
                                            return_dict=kwargs.get('return_dict')
                                            )
    
    def _prepare_encoder_inputs(self, image, input_ids=None, bbox=None, attention_mask=None):
        # text embedding
        batch_size = image.shape[0] 
        assert self.config.vision is not None
        # vision embedding
        vision_embeds = self.vision_model(image)
        assert self.vision_model.model_name == "swin_v2"
        vision_embeds = self.vision_embed_matcher(vision_embeds)
        vision_seq_length = vision_embeds.shape[1]
        # add task token (e.g <OCR> for ocr)
        vision_embeds, text_seq_length = self.concat_task_token(vision_embeds, 0)
        attention_mask = torch.ones((batch_size, vision_seq_length + text_seq_length), dtype=torch.int32).to(self.device)
        return None, vision_embeds, attention_mask

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        input_name = 'inputs_embeds'
        text_embeds, vision_embeds, attention_mask = self._prepare_encoder_inputs(input_ids=model_kwargs['questions']['input_ids'], image=model_kwargs['image'])
        model_kwargs['attention_mask'] = attention_mask
        inputs_embeds = vision_embeds
        inputs = inputs_embeds
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def add_task_tokens(self):
        super().add_task_tokens()
        self.input_tokenizer.add_tokens('<VQAP2S>', special_tokens=True)
        self.task_token_ids.update({'vqa_p2s': self.register_token('<VQAP2S>')})


def _init_weights_manual(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.2)
        if isinstance(m, nn.Linear) and m.bias is not None:
            trunc_normal_(m.bias, std=.5)


def _to_cuda(sample, device=torch.device('cuda')):
    if isinstance(sample, torch.Tensor):
        return sample.to(device)
    elif isinstance(sample, list):
        return sample
    else:
        for k in sample.keys():
            sample[k] = _to_cuda(sample[k], device)
        return sample


def fetch_sample(ds, ds_for_vis):
    idx = random.randint(50, 100)
    for i in range(idx):
        inputs = next(ds)
        inputs_to_vis = next(ds_for_vis)
    return inputs, inputs_to_vis


MATCHER_MAP = {
    'default': EmbedMatcher,
    'matcher_merger': EmbedMatcherMerger,
    'matcher_conv': EmbedMatcherConv,
    'matcher_conv_v2': EmbedMatcherConv_V2
}


# vqa
if __name__ == '__main__':
    import pickle
    from visfocus.utils.utils import get_model_class, get_last_checkpoint
    import logging
    import random
    from os.path import dirname


    with open('model_args.pkl', 'rb') as f:
        model_args = pickle.load(f)

    logger = logging.getLogger(__name__)

    model_args.model_config_path = 'checkpoints/fv_base_docvqa_v1/vqa_model_args.yaml'
    # model_args.model_config_path = 'checkpoints/vf_base_lmpm_pt/model_args.yaml'

    DEVICE = 'cuda' # 'cpu'

    ## load pretrained if needed
    last_ckpt = get_last_checkpoint(dirname(model_args.model_config_path))
    ##

    model = get_model_class(model_args, logger, last_ckpt='/mnt/efs/ofirab/TextractOCRFreeDFM/TextractOCRFreeDFM/textract_ocr_free_dfm/baselines/Deploy/VisFocus/checkpoints_pth/vf_base_docvqa_v1.pth')
    model.to(DEVICE)
