import os
import argparse
import torch
import datetime
import logging
from utils.misc import set_seed
from dataset import bio_bort_epi_whole_wikiann
from utils.bort_training import bort_train
from utils.bort_evaluation import bort_eval
from model.bort import BORT

def parse_args():
    """ Parse command line arguments and initialize experiment. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['bio_bort']
    )
    parser.add_argument(
        "--train_language",
        type=str,
        required=True,
        choices=['english']
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=128
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="Batch size to use for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
        help="Batch size to use for evaluation."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--weight_decay",
        default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--warmup_ratio",
        default=0.025, type=int, help="Linear warmup over warmup_ratio*total_steps.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Do training & validation."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Do Zero-shot validation."
    )
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        default=None,
        help="Model checkpoint path for evaluation."
    )
    args = parser.parse_args()
    args.start_time = datetime.datetime.now().strftime('%d-%m-%Y_%Hh%Mm%Ss')
    args.output_dir = os.path.join(
        'results',
        args.task,
        f'{args.start_time}__seed-{args.seed}')

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO)

    # Check for GPUs
    if torch.cuda.is_available():
        assert torch.cuda.device_count() == 1  # This script doesn't support multi-gpu
        args.device = torch.device("cuda")
        logging.info("Using GPU (`%s`)", torch.cuda.get_device_name(0))
    else:
        args.device = torch.device("cpu")
        logging.info("Using CPU")

    # Set random seed for reproducibility
    set_seed(seed_value=args.seed)

    return args

def main(args):
    ##### GPU DEVICE #####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##### MODEL #####
    if args.model == 'bio_bort':
        ##### Model - BIO BORT #####
        logging.info('Loading BIO Tagging BORT model')
        model = BORT(7)
    else:
        print(f"{args.model} is not in the model list.")
        raise NotImplementedError
    model.to(device)

    ##### whole model param #####
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Whole model number of params:', n_parameters)

    ##### Train #####
    # Log args
    logging.info('Using the following arguments for training:')
    for k, v in vars(args).items():
        logging.info("* %s: %s", k, v)
    if args.do_train:
        ##### Train Dataset Loading #####
        if args.model == 'bio_bort' and args.train_language == 'english':
            ##### BIO WikiAnn Epi English Dataset #####
            logging.info('Loading BIO BORT Epi WikiAnn English Dataset')
            train_dataset = {}
            train_dataset['train'] = bio_bort_epi_whole_wikiann.BIOBORTEpiWikiAnnDataset('english', 'train', args.max_seq_len)
            train_dataset['validation'] = bio_bort_epi_whole_wikiann.BIOBORTEpiWikiAnnDataset('english', 'validation', args.max_seq_len)
            train_dataset['test'] = bio_bort_epi_whole_wikiann.BIOBORTEpiWikiAnnDataset('english', 'test', args.max_seq_len)
        else:
            print(f"{args.train_language} is not defined!")

        f = open(args.task + '_log.txt', 'a')
        global_step, train_loss, best_val_metric, best_val_epoch, best_model_state_dict = bort_train(
            args=args,
            english_dataset = train_dataset,
            zeroshot_dataset  = train_dataset['test'],
            model=model,
            device=device,
            f=f
        )
        logging.info("global_step = %s, average training loss = %s", global_step, train_loss)
        logging.info("Best performance: Epoch=%d, Value=%s", best_val_epoch, best_val_metric)
    
    ##### Zero-shot Validation #####
    if args.do_predict:
        model.load_state_dict(args.model_ckpt_path)
        model.eval()

        zero_shot_lang_list = ['korean', 'spanish', 'turkmen', 'maori', 'somali', 'uyghur', 'sinhala', 'quechua',
                               'assyrian', 'kashubian', 'ilocano', 'kyrgyz', 'kinyarwanda', 'esperanto', 'khmer',
                               'amharic', 'maltese', 'tajik', 'yoruba', 'marathi', 'javanese', 'urdu', 'malay',
                               'cebuano', 'croatian', 'malayalam', 'telugu', 'uzbek', 'punjabi']

        for lang in zero_shot_lang_list:
            logging.info(f'Loading BIO BORT {lang} Dataset')
            zeroshot_dataset = bio_bort_epi_whole_wikiann.BIOBORTEpiWikiAnnDataset(lang, 'validation', args.max_seq_len)

            zeroshot_results, _ = bort_eval(
                args=args,
                eval_dataset=zeroshot_dataset,
                model=model,
                device=device
            )

            logging.info("***** " + lang + " Eval results *****")
            f.write("\n***** " + lang + " Eval results *****\n")
            f.write('F1 metric: ' + str(round(zeroshot_results['f1'], 4)) + '\n')
            logging.info("F1 metric = %s", str(round(zeroshot_results['f1'], 4)))
        f.close()

if __name__ == "__main__":
    main(parse_args())
