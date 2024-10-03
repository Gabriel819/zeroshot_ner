from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.nn as nn

class BORT(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("palat/bort")
        self.num_labels = num_classes
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask).encoder_last_hidden_state
        out = self.classifier(out)

        return out
