config = {
    "data_root": "../data",
    "run_name": "xphonebert_ko_nikl_debug",
    "task": 'ner', # ner, 
    "hf_model": "vinai/xphonebert-base",
    "pretrained": None,
    # "run_name": "xphonebert_en_ipa_char_pull_kor_mse",
    # "xphonebert_en_ipa_word_lr0.01", # _cls_token_w_collate
    # "pretrained": "/workspace/zeroshot-ner/ckpt/task2/mbert_ko_klue_char_space_token_fs1e-2/last.pt",
    "freeze_encoder": False,
    "pull_kor": False,
    "fewshot": False,
    # "/workspace/zeroshot-ner/ckpt/task2/xphonebert_en_ipa_cls/best.pt",
    # "/workspace/zeroshot-ner/ckpt/task2/xphonebert_ko_ipa_smaller_dataset/last.pt",
    # "/workspace/zeroshot-ner/ckpt/task2/eng_epi_test_3nes_no_lr_scheduler_resumed/epoch_110.pt",
    "shuffle_word_order": False,
    "shuffle_word_order_schedule": {"init": 1.0, "milestone": [], "ratios": []},
    # langs: conll / onto / nikl / mrl / 
    "src_lang": ["en"], # en, ko, de
    "tgt_lang": ["nikl_news"],
    "use_ipa": True,
    # "model": 'xphonebert', # choose btween char_cnn / xphonebert / bert(only task2) / canine / charformer
    "repr": "char", # word, char, subtok
    "valid_nes" : ["PER", "LOC", "ORG"], # ["PER", "LOC", "ORG"],
    "seed": 42,
    "space": False,
    "device": "cuda:0",
    "optimizer": 'sgd', # sgd, adamw, adam
    "lr": 5e-3,
    "lr_scheduler": None,
    "epochs": 100,
    "batch_size": 128,
    "method": None, # None(default model), concat_bi_ner
    "b_loss_lambda": 0.25,
    "half_loss_lambda": 0.25,
    "context_attn_head": None,
    "number_of_characters": 1969, # default=70, xphonebert=1969, epitran=?
    "extra_characters": [],
    "dropout_input": 0.2,
    "max_length": 128,
    # entity 길이 (in IPA) CharCNN 사용할 때 기준, filter size/pool size 고려해서 해야 함,,
    "input_vector": "embedding", # ["embedding", "onehot"]
    "eval_every": 5,
    "save_every": 10,
    "wandb": False
}