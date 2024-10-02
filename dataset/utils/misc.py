""" Miscellaneous utils. """
import random
import logging
import torch
import numpy as np
from segments import Profile, Tokenizer
import pandas as pd

def set_seed(seed_value):
    """ Sets the random seed to a given value. """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    logging.info("Random seed: %d", seed_value)

def list_all_output_ipas(map_csv):
    out = []
    for k, lst in map_csv.values:
        out.append(lst)
    return out

def rm_nan_from_ipas(ipas):
    isnan=False
    for i, a in enumerate(ipas):
        if isinstance(a, float):
            # a = i
            isnan = True
            break
    if isnan:
        ipas.remove(a)
    return ipas

def get_epitran_map(lang):
    if lang == 'spanish':
        file_code = 'spa-Latn'
    elif lang == 'maori':
        file_code = 'mri-Latn'
    elif lang == 'somali':
        file_code = 'som-Latn'
    elif lang == 'uyghur':
        file_code = 'uig-Arab'
    elif lang == 'sinhala':
        file_code = 'sin-Sinh'
    elif lang == 'quechua':
        file_code = 'quy-Latn'
    elif lang == 'assyrian':
        file_code = 'aii-Syrc'
    elif lang == 'kashubian':
        file_code = 'csb-Latn'
    elif lang == 'ilocano':
        file_code = 'ilo-Latn'
    elif lang == 'gan':
        file_code = 'gan-Latn'
    elif lang == 'kyrgyz':
        file_code = 'kir-Arab'
    elif lang == 'kinyarwanda':
        file_code = 'kin-Latn'
    elif lang == 'korean':
        file_code = 'kor-Hang'
    elif lang == 'esperanto':
        file_code = 'epo-Latn'
    elif lang == 'khmer':
        file_code = 'khm-Khmr'
    elif lang == 'turkmen':
        file_code = 'tuk-Latn'
    elif lang == 'amharic':
        file_code = 'amh-Ethi'
    elif lang == 'maltese':
        file_code = 'mlt-Latn'
    elif lang == 'tajik':
        file_code = 'tgk-Cyrl'
    elif lang == 'yoruba':
        file_code = 'yor-Latn'
    elif lang == 'marathi':
        file_code = 'mar-Deva'
    elif lang == 'javanese':
        file_code = 'jav-Latn'
    elif lang == 'urdu':
        file_code = 'urd-Arab'
    elif lang == 'malay':
        file_code = 'msa-Latn'
    elif lang == 'cebuano':
        file_code = 'ceb-Latn'
    elif lang == 'croatian':
        file_code = 'hrv-Latn'
    elif lang == 'malayalam':
        file_code = 'mal-Mlym'
    elif lang == 'telugu':
        file_code = 'tel-Telu'
    elif lang == 'uzbek':
        file_code = 'uzb-Latn'
    elif lang == 'punjabi':
        file_code = 'pan-Guru'
    else:
        print(f"{lang} is not Implemented in ipa map")
        raise NotImplementedError
    map_csv = pd.read_csv(f"../data/{file_code}.csv")
    return map_csv

def get_phoneme_segmenter(lang):
    if lang:
        all_ipas = []
        if lang != 'english': # There isn't eng-Latn map in epitran
            ipas = list_all_output_ipas(get_epitran_map(lang))
            ipas = rm_nan_from_ipas(ipas)
            all_ipas += ipas
        ipa_vocab = pd.read_csv("../data/ipa_all.csv")['ipa'].unique().tolist()
        add_chars = list("▁ ._-.,/?<>;:'\"\][|{}]!@#$%^&*()=+1234567890")
        graphemes = list(set(all_ipas + ipa_vocab + add_chars))
        graphemes = rm_nan_from_ipas(graphemes)
        prf_dict = (
            [{"Grapheme": grapheme} for grapheme in graphemes]
        )
        prf = Profile(*prf_dict)
        tokenizer = Tokenizer(profile=prf)
    else:
        tokenizer = Tokenizer()
    return tokenizer
