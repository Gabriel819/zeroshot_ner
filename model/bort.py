from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.nn as nn

class BORT(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # self.model = BartForConditionalGeneration.from_pretrained("palat/bort")
        self.model = BartForConditionalGeneration.from_pretrained("/jimin/huggingface/hub/models--palat--bort/snapshots/4d76e3ebf80e967148fd5ac81dc364e470083d76/", tf=True, local_files_only=True)
        self.num_labels = num_classes
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask): # input_ids & attention_mask: (b_s, max_seq)
        out = self.model(input_ids, attention_mask).encoder_last_hidden_state # (b_s, max_seq, num_classes)
        # out.encoder_last_hidden_state: (), out.logits: (1, 26, 51201)
        out = self.classifier(out)

        return out

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('palat/bort')
    text = "Due to its coastal location, Long ·b·iʧ winter temperatures are milder than most of the state."
    encoded_input = tokenizer(text, return_tensors='pt')

    model = BORT(7)

    # forward pass
    output = model(**encoded_input)
    print()
