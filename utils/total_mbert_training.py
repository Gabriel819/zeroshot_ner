import torch
import os
import logging
import tqdm
from torch.utils.data import RandomSampler, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from dataset import bio_mbert_whole
from utils.misc import set_seed
from transformers import BertTokenizer
from utils.mbert_evaluation import mbert_eval
import evaluate
import torch.nn as nn
import copy

def total_mbert_train(args, conll_dataset, zeroshot_dataset, model, device, f):
    train_dataset = conll_dataset['train']
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size)

    n_train_steps = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * n_train_steps),
        num_training_steps=n_train_steps
    )

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info(
        "  Total train batch size = %d",
        args.train_batch_size
    )
    logging.info("  Total optimization steps = %d", n_train_steps)
    logging.info("  Using linear warmup (ratio=%s)", args.warmup_ratio)
    logging.info("  Using weight decay (value=%s)", args.weight_decay)
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    best_metric, best_epoch, zeroshot_best_metric = -1.0, -1, -1.0  # Init best -1 so that 0 > best

    model.zero_grad()
    train_iterator = tqdm.trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
    
    cross_entropy_loss = nn.CrossEntropyLoss()
    f1_metric = evaluate.load("f1")
    out_label_list, preds_list = [], []

    best_model_state_dict = None

    set_seed(seed_value=args.seed)  # For reproductibility
    for num_epoch in train_iterator:
        epoch_iterator = tqdm.tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            orig_token_inputs = tokenizer(batch[0], padding='max_length', truncation=True,
                                          max_length=args.max_seq_len, return_tensors="pt").to(device)
            model.train()

            outputs = model(**orig_token_inputs)
            logits = torch.argmax(outputs, dim=2)

            loss = cross_entropy_loss(outputs.flatten(0,1), batch[1].flatten().to(device))

            preds = logits.detach().cpu().numpy()
            out_label_ids = batch[1].detach().cpu().numpy()

            for i in range(out_label_ids.shape[0]):
                for j in range(out_label_ids.shape[1]):
                    if out_label_ids[i, j] != -100:
                        out_label_list.append(out_label_ids[i, j])
                        preds_list.append(preds[i][j])

            loss.backward()
            logging.info("\nStep %d Train loss = %s", global_step, loss.item())
            f.write("Step " + str(global_step) + " Train loss: " + str(loss.item()) + '\n')

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            train_results = f1_metric.compute(predictions=preds_list, references=out_label_list, average='macro')

        logging.info("***** Train results *****")
        f.write("\n***** Train results *****\n")
        f.write("Train step: " + str(global_step) + ' Result\n')
        f.write('F1 metric: ' + str(round(train_results['f1'], 4)) + '\n')
        logging.info("F1 metric = %s", str(round(train_results['f1'], 4)))

        ##### English Validation set Evaluation #####
        eval_results, _ = mbert_eval(
                args=args,
                eval_dataset=conll_dataset["validation"],
                model=model,
                device=device
        )

        logging.info("***** " + args.task + " Eval results *****")
        f.write("\n***** " + args.task + " Eval results *****\n")
        f.write("Eval step: " + str(global_step) + ' Result\n')
        f.write('F1 metric: ' + str(round(eval_results['f1'], 4)) + '\n')
        logging.info("F1 metric = %s", str(round(eval_results['f1'], 4)))

        ##### Zero-shot Evaluation #####
        zeroshot_results, _ = mbert_eval(
                args=args,
                eval_dataset=zeroshot_dataset,
                model=model,
                device=device
        )

        logging.info(f"***** {args.task} results *****")
        f.write(f"\n***** {args.task} results *****\n")
        f.write(f"{args.task} step: {global_step} Result\n)")

        f.write('F1 metric: ' + str(round(zeroshot_results['f1'], 4)) + '\n')
        logging.info("F1 metric = %s", str(round(zeroshot_results['f1'], 4)))
        zeroshot_metric = round(zeroshot_results['f1'], 4)

        if zeroshot_metric > zeroshot_best_metric:
            zeroshot_best_metric = zeroshot_metric
            best_epoch = num_epoch
            best_model_state_dict = copy.deepcopy(model.state_dict())

            # Save model checkpoint
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            best_model_state_dict = model.state_dict()
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.task + ".pth"))
            logging.info("Saving best zeroshot model checkpoint to %s", args.output_dir)

    return global_step, tr_loss / global_step, zeroshot_best_metric, best_epoch, best_model_state_dict
