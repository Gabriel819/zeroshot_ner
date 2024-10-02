# zeroshot_ner
Official code for [Zero-Shot Cross-Lingual NER Using Phonemic Representations for Low-Resource Languages](https://arxiv.org/abs/2406.16030), Jimin Sohn*, Haeji Jung*, Alex Cheng, Jooeun Kang, Yilin Du, David R. Mortensen, EMNLP 2024 main

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

# Train
## 1. mBERT train

## 2. CANINE train

## 3. xPhoneBERT train

## 4. BORT train

## 5. XML RoBERTa train

# Inference
## 1. mBERT inference

## 2. CANINE inference

## 3. xPhoneBERT inference

## 4. BORT inference

## 5. XML RoBERTa inference
