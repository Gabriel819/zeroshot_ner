from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn

class XML_RoBERTa(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
        self.num_labels = num_classes
        self.model.lm_head.decoder = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask)

        return out
