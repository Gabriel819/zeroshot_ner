import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import json

class BIOXMLRoBERTaWholeDataset(Dataset):
    def __init__(self, lang, split, max_seq_len, ratio=100):
        super(BIOXMLRoBERTaWholeDataset, self).__init__()
        # High-resource languages
        if lang == 'english':
            self.file = json.load(open('./data/english_wikiann.json'))
        elif lang == 'spanish':
            self.file = json.load(open('./data/spanish_wikiann.json'))
        elif lang == 'korean':
            self.file = json.load(open('./data/korean_wikiann.json'))
        # CASE 1 - 8 languages
        elif lang == 'maori':
            self.file = json.load(open('./data/maori_wikiann.json'))
        elif lang == 'somali':
            self.file = json.load(open('./data/somali_wikiann.json'))
        elif lang == 'uyghur':
            self.file = json.load(open('./data/uyghur_wikiann.json'))
        elif lang == 'sinhala':
            self.file = json.load(open('./data/sinhala_wikiann.json'))
        elif lang == 'quechua':
            self.file = json.load(open('./data/quechua_wikiann.json'))
        elif lang == 'assyrian':
            self.file = json.load(open('./data/assyrian_wikiann.json'))
        elif lang == 'ilocano':
            self.file = json.load(open('./data/ilocano_wikiann.json'))
        elif lang == 'kinyarwanda':
            self.file = json.load(open('./data/kinyarwanda_wikiann.json'))
        # CASE 2 - 12 languages
        elif lang == 'esperanto':
            self.file = json.load(open('./data/esperanto_wikiann.json'))
        elif lang == 'khmer':
            self.file = json.load(open('./data/khmer_wikiann.json'))
        elif lang == 'turkmen':
            self.file = json.load(open('./data/turkmen_wikiann.json'))
        elif lang == 'amharic':
            self.file = json.load(open('./data/amharic_wikiann.json'))
        elif lang == 'maltese':
            self.file = json.load(open('./data/maltese_wikiann.json'))
        elif lang == 'oriya':
            self.file = json.load(open('./data/oriya_wikiann.json'))
        elif lang == 'sanskrit':
            self.file = json.load(open('./data/sanskrit_wikiann.json'))
        elif lang == 'interlingua':
            self.file = json.load(open('./data/interlingua_wikiann.json'))
        elif lang == 'guarani':
            self.file = json.load(open('./data/guarani_wikiann.json'))
        elif lang == 'belarusian':
            self.file = json.load(open('./data/belarusian_wikiann.json'))
        elif lang == 'kurdish':
            self.file = json.load(open('./data/kurdish_wikiann.json'))
        elif lang == 'sindhi':
            self.file = json.load(open('./data/sindhi_wikiann.json'))
        # CASE 3 - 13 languages
        elif lang == 'tajik':
            self.file = json.load(open('./data/tajik_wikiann.json'))
        elif lang == 'yoruba':
            self.file = json.load(open('./data/yoruba_wikiann.json'))
        elif lang == 'marathi':
            self.file = json.load(open('./data/marathi_wikiann.json'))
        elif lang == 'javanese':
            self.file = json.load(open('./data/javanese_wikiann.json'))
        elif lang == 'urdu':
            self.file = json.load(open('./data/urdu_wikiann.json'))
        elif lang == 'malay':
            self.file = json.load(open('./data/malay_wikiann.json'))
        elif lang == 'cebuano':
            self.file = json.load(open('./data/cebuano_wikiann.json'))
        elif lang == 'croatian':
            self.file = json.load(open('./data/croatian_wikiann.json'))
        elif lang == 'malayalam':
            self.file = json.load(open('./data/malayalam_wikiann.json'))
        elif lang == 'telugu':
            self.file = json.load(open('./data/telugu_wikiann.json'))
        elif lang == 'uzbek':
            self.file = json.load(open('./data/uzbek_wikiann.json'))
        elif lang == 'punjabi':
            self.file = json.load(open('./data/punjabi_wikiann.json'))
        elif lang == 'kyrgyz':
            self.file = json.load(open('./data/kyrgyz_wikiann.json'))
        else:
            print(f"Language {lang} is not defined!")
            raise NotImplementedError
        self.lang = lang
        self.split = split
        self.max_seq_len = max_seq_len
        if split == 'train':
            total_len = len(self.file['train'])
            if ratio == 100:
                self.data = self.file['train']
            elif ratio == 50:
                cur_len = int(total_len * 0.5)
                self.data = self.file['train'][:cur_len]
            elif ratio == 30:
                cur_len = int(total_len * 0.3)
                self.data = self.file['train'][:cur_len]
        elif split == 'validation':
            self.data = self.file['validation']
        elif split == 'test':
            self.data = self.file['test']
        else:
            print("Dataset should be train or validation or test!")
            raise NotImplementedError
        self.tokens_list = [" ".join(ele['tokens']) for ele in self.data]
        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        self.pad_token = self.tokenizer.pad_token
        self.start_token = self.tokenizer.bos_token
        self.end_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        orig_tokens = cur_data['tokens']
        orig_ner_tags = cur_data['ner_tags']
        tokens, token_ner_tags = [], []

        ##### original token pre-process #####
        for i, ele in enumerate(orig_tokens):
            cur_input_ids = self.tokenizer(ele, return_tensors="pt")
            cur_ner_tag = orig_ner_tags[i]
            len_input_ids = cur_input_ids['input_ids'].shape[1]
            if len_input_ids == 3:
                token_ner_tags += [cur_ner_tag]
            else:  # one word becomes more than one token
                if cur_ner_tag in [1, 3, 5]:  # begin tag
                    tmp_ner_tag = [cur_ner_tag] + [cur_ner_tag + 1] * (len_input_ids - 3)
                else:
                    tmp_ner_tag = [cur_ner_tag] * (len_input_ids - 2)
                token_ner_tags += tmp_ner_tag

        # add CLS and SEP token at the ner tags
        token_ner_tags = [0] + token_ner_tags + [0]

        if self.max_seq_len > len(token_ner_tags):
            token_ner_tags += [-100] * (self.max_seq_len - len(token_ner_tags))
        elif self.max_seq_len < len(token_ner_tags):
            token_ner_tags = token_ner_tags[:self.max_seq_len]

        return " ".join(orig_tokens), torch.tensor(token_ner_tags)
