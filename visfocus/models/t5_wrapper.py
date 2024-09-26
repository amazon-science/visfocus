from torch import nn
from transformers import T5Tokenizer, T5Model
from peft import LoraModel, LoraConfig


class VisionModel(nn.Module):
    """ this should be your smart order model (takes an image as input)
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.model_name = 'vision_model'
        self.model = nn.Identity()

    def forward(self, x):
        return self.model(x)


class T5_wrapper(nn.Module):
    def __init__(self, t5_variant='small', lora_conf=None,freeze=False, logger=None):
        super().__init__()
        self.model_arch = 't5_wrapper'
        self.input_tokenizer = self.output_tokenizer = self.language_model_tokenizer = T5Tokenizer.from_pretrained(f't5-{t5_variant}')
        self.language_model_tokenizer.generate_max_new_tokens_len = 2048 # long for ocr
        self.model = T5Model.from_pretrained(f't5-{t5_variant}')
        self.log_fn = logger.warning if logger else print
        self.generation_config = self.model.generation_config

        self.vision_model = VisionModel()

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        
        if lora_conf:
            self.apply_lora(lora_conf)
            
    def apply_lora(self, lora_conf):
        lora_conf = LoraConfig(**lora_conf)
        self.model = LoraModel(self.model, lora_conf, 'default').model
        _lm_total = sum(p.numel() for p in self.parameters())
        _lm_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.log_fn(f'LoRa uses {_lm_trainable} / {_lm_total} trainable params')

    def forward(self, *args, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            return_dict=True,
        )
        return outputs[0]

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


if __name__ == '__main__':
      lora_conf = {
        'task_type': 'CAUSAL_LM',
        'r': 16,
        'lora_alpha': 32,
        'bias': 'none',
        'lora_dropout': 0.01,
        'target_modules': ['q', 'v']
      }
      
      T5_wrapper(lora_conf=lora_conf)
