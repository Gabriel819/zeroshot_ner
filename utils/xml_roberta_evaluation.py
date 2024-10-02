import torch
import numpy as np
import logging
from torch.utils.data import SequentialSampler, DataLoader
import tqdm
from transformers import AutoTokenizer
import evaluate
import torch.nn as nn

def xml_roberta_eval(args, eval_dataset, model, device):
    """ Evaluates the given model on the given dataset. """
    f1_metric = evaluate.load("f1")
    
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size)
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    
    # Evaluation
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
        orig_token_inputs = tokenizer(batch[0], padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors="pt").to(device)
        outputs = model(**orig_token_inputs)
        outputs = outputs.logits
        logits = torch.argmax(outputs, dim=2)

        loss = cross_entropy_loss(outputs.flatten(0, 1), batch[1].flatten().to(device))

        eval_loss += loss.item()
        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()
        out_label_ids = batch[1].detach().cpu().numpy()

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != -100:
                    out_label_list.append(out_label_ids[i, j])
                    preds_list.append(preds[i][j])

    results = f1_metric.compute(predictions=preds_list, references=out_label_list, average='macro')

    return results, preds_list
