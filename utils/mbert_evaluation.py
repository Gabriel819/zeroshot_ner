import torch
import numpy as np
import logging
from torch.utils.data import SequentialSampler, DataLoader
import tqdm
from transformers import BertTokenizer
import evaluate
import torch.nn as nn

def mbert_eval(args, eval_dataset, model, device):
    """ Evaluates the given model on the given dataset. """
    # f1_metric = evaluate.load("f1")
    f1_metric = evaluate.load("/jimin/huggingface/modules/evaluate_modules/metrics/evaluate-metric--f1/0ca73f6cf92ef5a268320c697f7b940d1030f8471714bffdb6856c641b818974/f1.py")

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size)
    # tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
    tokenizer = BertTokenizer.from_pretrained("/jimin/huggingface/hub/models--google-bert--bert-base-multilingual-cased/snapshots/3f076fdb1ab68d5b2880cb87a0886f315b8146f8")

    # Evaluate!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    out_label_list, preds_list = [], []

    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)

    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
        # orig_token_inputs = tokenizer(batch[0], add_special_tokens=False, padding='max_length',
        #                                   truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
        orig_token_inputs = tokenizer(batch[0], padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors="pt").to(device)
        outputs = model(**orig_token_inputs)  # outputs: (b_s, max_len, class_num)
        logits = torch.argmax(outputs, dim=2)  # logits: (b_s, max_len)

        loss = cross_entropy_loss(outputs.flatten(0, 1), batch[2].flatten().to(device))

        eval_loss += loss.item()
        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()
        # preds = np.argmax(preds, axis=2) # (128, 256)
        out_label_ids = batch[2].detach().cpu().numpy() # (128, 256)

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != -100:  # 0 is pad_token_id
                    out_label_list.append(out_label_ids[i, j])
                    preds_list.append(preds[i][j])

    results = f1_metric.compute(predictions=preds_list, references=out_label_list, average='macro')

    return results, preds_list
