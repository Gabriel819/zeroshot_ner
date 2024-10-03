from transformers import CanineModel
import torch.nn as nn

class CANINE_C(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = CanineModel.from_pretrained('google/canine-c')
        self.num_labels = num_classes
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.model(input_ids, attention_mask, token_type_ids)
        x = x.last_hidden_state
        out = self.classifier(x)

        return out
