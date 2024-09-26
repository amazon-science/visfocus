from torch import nn
from transformers import T5Tokenizer, T5Model


class T5_Encoder(nn.Module):
    def __init__(self, t5_variant='small', freeze=True):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(f'{t5_variant}')
        model = T5Model.from_pretrained(f'{t5_variant}')
        del model.decoder
        self.encoder = model.encoder
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            return_dict=True,
        )
        return encoder_outputs[0]