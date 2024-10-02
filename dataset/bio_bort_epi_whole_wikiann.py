import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from utils.misc import get_phoneme_segmenter
from datasets import load_dataset

class BIOBORTEpiWikiAnnDataset(Dataset):
    def __init__(self, lang, split, max_seq_len):
        super(BIOBORTEpiWikiAnnDataset, self).__init__()
        if lang == 'english':
            self.file = json.load(open('./data/real_english_wikiann.json'))
        elif lang == 'spanish':
            self.file = json.load(open('./data/real_spanish_wikiann.json'))
        elif lang == 'korean':
            self.file = json.load(open('./data/real_korean_wikiann.json'))
        # CASE 1
        elif lang == 'maori':
            self.file = json.load(open('./data/real_maori_wikiann.json'))
        elif lang == 'somali':
            self.file = json.load(open('../data/real_somali_wikiann.json'))
        elif lang == 'uyghur':
            self.file = json.load(open('./data/real_uyghur_wikiann.json'))
        elif lang == 'sinhala':
            self.file = json.load(open('../data/real_sinhala_wikiann.json'))
        elif lang == 'quechua':
            self.file = json.load(open('./data/real_quechua_wikiann.json'))
        elif lang == 'assyrian':
            self.file = json.load(open('../data/real_assyrian_wikiann.json'))
        elif lang == 'kashubian':
            self.file = json.load(open('./data/real_kashubian_wikiann.json'))
        elif lang == 'ilocano':
            self.file = json.load(open('./data/real_ilocano_wikiann.json'))
        elif lang == 'kyrgyz':
            self.file = json.load(open('./data/final_kyrgyz_wikiann.json'))
        elif lang == 'kinyarwanda':
            self.file = json.load(open('./data/real_kinyarwanda_wikiann.json'))
        # CASE 2
        elif lang == 'esperanto':
            self.file = json.load(open('./data/real_esperanto_wikiann.json'))
        elif lang == 'khmer':
            self.file = json.load(open('./data/real_khmer_wikiann.json'))
        elif lang == 'turkmen':
            self.file = json.load(open('./data/real_turkmen_wikiann.json'))
        elif lang == 'amharic':
            self.file = json.load(open('./data/real_amharic_wikiann.json'))
        elif lang == 'maltese':
            self.file = json.load(open('./data/real_maltese_wikiann.json'))
        # CASE 3
        elif lang == 'tajik':
            self.file = json.load(open('./data/real_tajik_wikiann.json'))
        elif lang == 'yoruba':
            self.file = json.load(open('./data/real_yoruba_wikiann.json'))
        elif lang == 'marathi':
            self.file = json.load(open('./data/real_marathi_wikiann.json'))
        elif lang == 'javanese':
            self.file = json.load(open('./data/real_javanese_wikiann.json'))
        elif lang == 'urdu':
            self.file = json.load(open('./data/real_urdu_wikiann.json'))
        elif lang == 'malay':
            self.file = json.load(open('./data/real_malay_wikiann.json'))
        elif lang == 'cebuano':
            self.file = json.load(open('./data/real_cebuano_wikiann.json'))
        elif lang == 'croatian':
            self.file = json.load(open('./data/real_croatian_wikiann.json'))
        elif lang == 'malayalam':
            self.file = json.load(open('./data/real_malayalam_wikiann.json'))
        elif lang == 'telugu':
            self.file = json.load(open('./data/real_telugu_wikiann.json'))
        elif lang == 'uzbek':
            self.file = json.load(open('./data/real_uzbek_wikiann.json'))
        elif lang == 'punjabi':
            self.file = json.load(open('./data/real_punjabi_wikiann.json'))
        # Additional
        elif lang == 'oriya':
            self.file = json.load(open('./data/t2ps_oriya_wikiann.json'))
        elif lang == 'belarusian':
            self.file = json.load(open('./data/t2ps_belarusian_wikiann.json'))
        elif lang == 'kurdish':
            self.file = json.load(open('./data/t2ps_kurdish_wikiann.json'))
        elif lang == 'sanskrit':
            self.file = json.load(open('./data/t2ps_sanskrit_wikiann.json'))
        elif lang == 'interlingua':
            self.file = json.load(open('./data/t2ps_interlingua_wikiann.json'))
        elif lang == 'guarani':
            self.file = json.load(open('./data/t2ps_guarani_wikiann.json'))
        else:
            print(f"Language {lang} is not defined!")
            raise NotImplementedError
        self.lang = lang
        # {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        self.split = split
        self.max_seq_len = max_seq_len
        if split == 'train':
            self.data = self.file['train']
        elif split == 'validation':
            self.data = self.file['validation']
        elif split == 'test':
            self.data = self.file['test']
        else:
            print("Dataset should be train or validation or test!")
            raise NotImplementedError
        self.tokenizer = AutoTokenizer.from_pretrained("palat/bort")
        self.pad_token = self.tokenizer.pad_token
        self.start_token = self.tokenizer.bos_token
        self.end_token = self.tokenizer.eos_token
        self.ipa_tokenizer = get_phoneme_segmenter(lang)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cur_data = self.data[idx]
        orig_epi_tokens = cur_data['epi_tokens']
        orig_epi_ner_tags = cur_data['ner_tags']
        epi_tokens, epi_ner_tags = [], []

        # orig_tokens = ['John', 'works', 'for', 'Globex', 'Corp.', '.']
        # orig_ner_tags = [1, 0, 0, 3, 4, 0]
        # orig_tokens = ['John', 'works', 'for', 'Globex', 'Corp.', 'with', 'Paul', '.']
        # orig_ner_tags = [1,0,0,3,4,0,1,0]

        ##### Epi token pre-process #####
        for i, ele in enumerate(orig_epi_tokens):
            tmp_ipa = self.ipa_tokenizer(ele)
            epi_tokens.append(tmp_ipa)
            ch_input_ids = self.tokenizer(tmp_ipa, return_tensors="pt")
            real_ner = orig_epi_ner_tags[i]
            len_ch_input_ids = len(ch_input_ids['input_ids'][0])
            if len_ch_input_ids == 3:
                epi_ner_tags += [real_ner]
            else:
                if real_ner in [1, 3, 5]:
                    epi_ner_tags += [real_ner]
                    epi_ner_tags += [real_ner + 1] * (len_ch_input_ids - 3)
                else:
                    epi_ner_tags += [real_ner] * (len_ch_input_ids - 2)
            # space's ner tag
            if i != len(orig_epi_tokens) - 1:
                epi_ner_tags += [0]

        epi_ner_tags = [0] + epi_ner_tags + [0]  # bos and eos tagging

        if self.max_seq_len > len(epi_ner_tags):
            epi_ner_tags += [-100] * (self.max_seq_len - len(epi_ner_tags))
        elif self.max_seq_len < len(epi_ner_tags):
            epi_ner_tags = epi_ner_tags[:self.max_seq_len]

        # return " ▁ ".join(epi_tokens), " ▁ ".join(epi_tokens), torch.tensor(epi_ner_tags), torch.tensor(epi_ner_tags)
        return "·".join(epi_tokens), "·".join(epi_tokens), torch.tensor(epi_ner_tags), torch.tensor(epi_ner_tags)

if __name__ == '__main__':
    dataset = BIOBORTEpiWikiAnnDataset('somali', 'validation', 128)
    tmp = next(iter(dataset)) # sample data
    # a: list 128, b: list 128, c: tensor (128,), d: tensor (128,)

    tokenizer = AutoTokenizer.from_pretrained("palat/bort")
    input_ids = tokenizer(tmp[0], return_tensors="pt")

    print()