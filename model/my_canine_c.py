from transformers import CanineModel
import torch.nn as nn

class My_CANINE_C(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # self.model = CanineModel.from_pretrained('google/canine-c')
        self.model = CanineModel.from_pretrained("/jimin/huggingface/hub/models--google--canine-c/snapshots/59de14c6de8e676db78de601ac4cb470b56d4d25", local_files_only=True)

        self.num_labels = num_classes
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids): # input_ids&attention_mask&token_type_ids: (b_s, max_seq)
        x = self.model(input_ids, attention_mask, token_type_ids)
        x = x.last_hidden_state # x: (b_s, max_seq, 768)
        out = self.classifier(x) # out: (b_s, max_seq, 7)

        return out
