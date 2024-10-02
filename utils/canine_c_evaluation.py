import torch
import numpy as np
import logging
from torch.utils.data import SequentialSampler, DataLoader
import tqdm
from transformers import CanineTokenizer
import evaluate

def total_canine_c_eval(args, eval_dataset, model, device):
    """ Evaluates the given model on the given dataset. """
    f1_metric = evaluate.load("f1")
    
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size)
    tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
    cross_ent = torch.nn.CrossEntropyLoss()

    # Evaluate!
    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    out_label_list, preds_list = [], []
    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating"):
        orig_token_inputs = tokenizer(batch[0], padding='max_length',
                                      truncation=True, max_length=args.max_seq_len, return_tensors="pt").to(device)
        with torch.no_grad():
            orig_token_inputs = {
                "input_ids": orig_token_inputs.data['input_ids'].to(device),
                "attention_mask": orig_token_inputs.data['attention_mask'].to(device),
                "token_type_ids": orig_token_inputs.data['token_type_ids'].to(device)
            }
            logits = model(**orig_token_inputs)
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

    results = f1_metric.compute(predictions=preds_list, references=out_label_list, average='macro')

    return results, preds_list
