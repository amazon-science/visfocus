from torch import nn
from transformers import AutoModel, AutoTokenizer


class BERT(nn.Module):
    def __init__(self, variant='tiny', freeze=True):
        super().__init__()
        model_name = f'prajjwal1/bert-{variant}'
        self.tokenizer = AutoTokenizer.from_pretrained(f'{model_name}')
        self.bert = AutoModel.from_pretrained(f'{model_name}')
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids):
        orig_text = self.orig_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        converted_input_ids = self.tokenizer(orig_text, padding=True, max_length=512, truncation=True, return_tensors='pt').input_ids.to(self.bert.device)
        outputs = self.bert(
            input_ids=converted_input_ids,
            return_dict=True,
        )
        return outputs.last_hidden_state