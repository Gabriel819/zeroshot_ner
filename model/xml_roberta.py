from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn

class XML_RoBERTa(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # self.model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")
        self.model = AutoModelForMaskedLM.from_pretrained("/jimin/huggingface/hub/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089", local_files_only=True)
        self.num_labels = num_classes
        self.model.lm_head.decoder = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask): # input_ids & attention_mask: (b_s, max_seq)
        out = self.model(input_ids, attention_mask) # (b_s, max_seq, num_classes)

        return out

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')

    model = XML_RoBERTa(7)

    # forward pass
    output = model(**encoded_input)
    print()
