# zeroshot_ner
Official code for [Zero-Shot Cross-Lingual NER Using Phonemic Representations for Low-Resource Languages](https://aclanthology.org/2024.emnlp-main.753/),

Jimin Sohn*, Haeji Jung*, Alex Cheng, Jooeon Kang, Yilin Du, David R. Mortensen, EMNLP 2024 main

*: equal contribution

# Approach
<img width="548" alt="image" src="https://github.com/user-attachments/assets/6c8c95f0-9a65-49e4-8feb-3fc85afa6196">

# Data Structure
```
data
├── id_mapping                   # 0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'
├── validation                   # Validation split
    ├── tokens                   # Tokens for mBERT, CANINE, XML-RoBERTa
    ├── ner_tags                 # BIO NER tags for mBERT, CANINE, XML-RoBERTa
    ├── epi_tokens               # IPA Tokens for XPhoneBERT, BORT
    └── epi_ner_tags             # BIO IPA NER tags for XPhoneBERT, BORT
└── train                        # Train split
    ├── tokens                   
    ├── ner_tags
    ├── epi_tokens
    └── epi_ner_tags
```

# Data
<img width="748" alt="image" src="https://github.com/user-attachments/assets/e23687ea-1a2a-40c8-a28a-4b282b4391a8">

- M: mBERT, C: CANINE, X: XPhoneBERT
- V represents the languages pre-trained on the model.

# Train
```
1. mBERT train
python src/mbert_main.py --task mbert --model bio_mbert --train_language english --max_seq_len 128 --do_train --train_batch_size 128 --eval_batch_size 128 --num_train_epochs 10

2. CANINE train
python src/canine_c_main.py --task canine_c --model bio_canine_ch --train_language english --max_seq_len 128 --do_train --train_batch_size 128 --eval_batch_size 128 --num_train_epochs 10

3. XPhoneBERT train
python src/xphonebert_main.py --task phoneme_xphonebert --model bio_xphonebert --train_language english --max_seq_len 128 --do_train --train_batch_size 128 --eval_batch_size 128 --num_train_epochs 10

4. BORT train
python src/bort_main.py --task phoneme_bort --model bio_bort --train_language english --max_seq_len 128 --do_train --train_batch_size 128 --eval_batch_size 128 --num_train_epochs 10

5. XML RoBERTa train
python src/xml_roberta_main.py --task xml_roberta --model bio_xml_roberta --train_language english --max_seq_len 128 --do_train --train_batch_size 128 --eval_batch_size 128 --num_train_epochs 10
```
# Inference
```
1. mBERT inference
python src/mbert_main.py --task mbert --model bio_mbert --max_seq_len 128 --do_predict --eval_batch_size 128 --model_ckpt_path 'mbert model checkpoint path'

2. CANINE inference
python src/canine_c_main.py --task canine_c --model bio_canine_ch --max_seq_len 128 --do_predict --eval_batch_size 128 --model_ckpt_path 'canine c model checkpoint path'

3. XPhoneBERT inference
python src/xphonebert_main.py --task phoneme_xphonebert --model bio_xphonebert --max_seq_len 128 --do_predict --eval_batch_size 128 --model_ckpt_path 'xphonebert model checkpoint path'

4. BORT inference
python src/bort_main.py --task phoneme_bort --model bio_bort --max_seq_len 128 --do_predict --eval_batch_size 128 --model_ckpt_path 'bort model checkpoint path'

5. XML RoBERTa inference
python src/xml_roberta_main.py --task xml_roberta --model bio_xml_roberta --max_seq_len 128 --do_predict --eval_batch_size 128 --model_ckpt_path 'xml roberta model checkpoint path'
```

# Email
estelle26598@gm.gist.ac.kr, gpwl0709@korea.ac.kr

# Cite
```
@inproceedings{sohn-etal-2024-zero,
    title = "Zero-Shot Cross-Lingual {NER} Using Phonemic Representations for Low-Resource Languages",
    author = "Sohn, Jimin  and
      Jung, Haeji  and
      Cheng, Alex  and
      Kang, Jooeon  and
      Du, Yilin  and
      Mortensen, David R",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.753",
    pages = "13595--13602",
    abstract = "Existing zero-shot cross-lingual NER approaches require substantial prior knowledge of the target language, which is impractical for low-resource languages.In this paper, we propose a novel approach to NER using phonemic representation based on the International Phonetic Alphabet (IPA) to bridge the gap between representations of different languages.Our experiments show that our method significantly outperforms baseline models in extremely low-resource languages, with the highest average F1 score (46.38{\%}) and lowest standard deviation (12.67), particularly demonstrating its robustness with non-Latin scripts. Ourcodes are available at https://github.com/Gabriel819/zeroshot{\_}ner.git",
}
```
