from typing import Optional

import torch
from transformers import AutoConfig, PreTrainedModel, AutoModel

from transformer import TransformerBlock

class Model(PreTrainedModel):

    def __init__(self, path, linear_probe=False, num_label=2, dropout=0.1):
        config = AutoConfig.from_pretrained(path)
        super().__init__(config)
        self.backbone_model = AutoModel.from_pretrained(path)
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))
        print(hidden_size)
        n_head = 8
        self.transformers = TransformerBlock(d_model=hidden_size, n_heads=n_head)
        self.score = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_size, num_label)
        )
        
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        )

    def forward(self, 
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.Tensor] = None):
        transformer_outputs = self.backbone_model(input_ids, attention_mask=attention_mask)
        
        hidden_states = self.transformers(transformer_outputs.last_hidden_state)    # torch.Size([8, 499, 1024])    # torch.Size([8, 499, 1024])
        # print(hidden_states.shape)
        hidden_states = torch.mean(hidden_states, dim=-2)   # torch.Size([8, 1024])
        
        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        return logits
