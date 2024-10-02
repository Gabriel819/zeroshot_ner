import torch
import numpy as np
import logging
from torch.utils.data import SequentialSampler, DataLoader
import tqdm
import sklearn.metrics as sklearn_metrics
from transformers import AutoTokenizer
import evaluate

def xphonebert_eval(args, eval_dataset, model, device):
    """ Evaluates the given model on the given dataset. """
    f1_metric = evaluate.load("f1")
    
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size)
    tokenizer = AutoTokenizer.from_pretrained("vinai/xphonebert-base")
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
        orig_epi_token_inputs = tokenizer(list(batch[0]), return_tensors="pt", padding='max_length',
                                          truncation=True, max_length=args.max_seq_len)

        with torch.no_grad():
            epi_token_inputs = orig_epi_token_inputs['input_ids'].to(device) # (b_s, max_seq)
            epi_attn_mask = orig_epi_token_inputs['attention_mask'].to(device)

            outputs = model(epi_token_inputs, attn_mask=epi_attn_mask)
            logits = outputs['logits']
            label = batch[1].to(device)
            tmp_eval_loss = cross_ent(logits.flatten(0, 1), label.flatten())

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2)
        out_label_ids = batch[1].detach().cpu().numpy()

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != -100:
                    out_label_list.append(out_label_ids[i, j])
                    preds_list.append(preds[i][j])

    eval_loss = eval_loss / nb_eval_steps

    results = f1_metric.compute(predictions=preds_list, references=out_label_list, average='macro')

    return results, preds_list
