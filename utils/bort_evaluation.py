import torch
import numpy as np
import logging
from torch.utils.data import SequentialSampler, DataLoader
import tqdm
import sklearn.metrics as sklearn_metrics
from transformers import AutoTokenizer
import evaluate

def bort_eval(args, eval_dataset, model, device):
    """ Evaluates the given model on the given dataset. """
    # f1_metric = evaluate.load("f1")
    f1_metric = evaluate.load("/jimin/huggingface/modules/evaluate_modules/metrics/evaluate-metric--f1/0ca73f6cf92ef5a268320c697f7b940d1030f8471714bffdb6856c641b818974/f1.py")

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size)
    # tokenizer = AutoTokenizer.from_pretrained("palat/bort")
    tokenizer = AutoTokenizer.from_pretrained("/jimin/huggingface/hub/models--palat--bort/snapshots/4d76e3ebf80e967148fd5ac81dc364e470083d76", local_files_only=True)
    cross_ent = torch.nn.CrossEntropyLoss(ignore_index=-100)

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
    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
        orig_epi_token_inputs = tokenizer(list(batch[1]), return_tensors="pt", padding='max_length',
                                          truncation=True, max_length=args.max_seq_len)

        with torch.no_grad():
            epi_token_inputs = orig_epi_token_inputs['input_ids'].to(device) # (b_s, max_seq)
            epi_attn_mask = orig_epi_token_inputs['attention_mask'].to(device)

            outputs = model(epi_token_inputs, attn_mask=epi_attn_mask)
            # outputs.feature: tensor (b_s, max_seq, 768), outputs.logits: (b_s, max_seq, 7)
            logits = outputs['logits']  # logits: (b_s, max_seq, 7)
            label = batch[3].to(device)  # (batch_size, max_seq_len)
            tmp_eval_loss = cross_ent(logits.flatten(0, 1), label.flatten())  # (64, 256, 7), (64, 256)

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2) # (128, 256)
        out_label_ids = batch[3].detach().cpu().numpy() # (128, 256)
        # id2ner = {0:'O', 1:'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', 5:'B-LOC', 6:'I-LOC'}

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != -100:  # 0 is pad_token_id
                    out_label_list.append(out_label_ids[i, j])
                    preds_list.append(preds[i][j])

    eval_loss = eval_loss / nb_eval_steps
    results = f1_metric.compute(predictions=preds_list, references=out_label_list, average='macro')

    return results, preds_list
