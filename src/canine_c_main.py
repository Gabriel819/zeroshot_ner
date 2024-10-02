import os
import argparse
import torch
import datetime
import logging
from utils.misc import set_seed
from dataset import bio_canine_ch_whole_wikiann, bio_canine_epi_whole_wikiann
from utils.canine_c_training import canine_c_train
from utils.canine_c_evaluation import canine_c_eval
from model.my_canine_c import My_CANINE_C

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
        choices=['bio_canine_ch', 'bio_canine_epi']
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
        help="Do prediction on the test set."
    )
    parser.add_argument(
        "--do_zero_predict",
        action="store_true",
        help="Do prediction on the zero-shot language validation set."
    )
    parser.add_argument(
        "--do_additional_predict",
        action="store_true",
        help="Do additional prediction on the zero-shot language validation set."
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
    if args.model == 'bio_canine_ch' or args.model == 'bio_canine_epi':
        ##### Model - CANINE-C #####
        logging.info('Loading CANINE-C model')
        model = My_CANINE_C(7)
    else:
        print("Wrong Model input")
        raise NotImplementedError
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # whole model param#
    print('Whole model number of params:', n_parameters)

    ##### Train #####
    # Log args
    logging.info('Using the following arguments for training:')
    for k, v in vars(args).items():
        logging.info("* %s: %s", k, v)
    if args.do_train:
        ##### Train Dataset Loading #####
        if args.model == 'bio_canine_ch' and args.train_language == 'english':  # for now, just use eng wikiann for training
            ##### BIO WikiAnn ch English Dataset #####
            logging.info('Loading BIO CANINE ch WikiAnn English Dataset')
            train_dataset = {}
            train_dataset['train'] = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset('english', 'train',
                                                                                           args.max_seq_len)
            train_dataset['validation'] = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset('english', 'validation',
                                                                                                args.max_seq_len)
            train_dataset['test'] = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset('english', 'test',
                                                                                          args.max_seq_len)
        elif args.model == 'bio_canine_epi' and args.train_language == 'english':  # for now, just use eng wikiann for training
            ##### BIO WikiAnn epi English Dataset #####
            logging.info('Loading BIO CANINE epi WikiAnn English Dataset')
            train_dataset = {}
            train_dataset['train'] = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset('english', 'train',
                                                                                             args.max_seq_len)
            train_dataset['validation'] = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset('english',
                                                                                                  'validation',
                                                                                                  args.max_seq_len)
            train_dataset['test'] = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset('english', 'test',
                                                                                            args.max_seq_len)
        else:
            print(f"{args.model} is not defined!")

        ##### Zero-shot Test Dataset Loading #####
        ########## BIO CANINE ch ##########
        if args.model == 'bio_canine_ch':
            ##### BIO ch WikiAnn Korean Dataset #####
            logging.info('Loading BIO CANINE ch WikiAnn Korean Dataset')
            validation_dataset = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset('korean', 'validation',
                                                                                       args.max_seq_len)
        ########## BIO CANINE Epi ##########
        elif args.model == 'bio_canine_epi':
            ##### BIO Epi WikiAnn Korean Dataset #####
            logging.info('Loading BIO CANINE Epi WikiAnn Korean Dataset')
            validation_dataset = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset('korean', 'validation',
                                                                                         args.max_seq_len)
        else:
            print(f"{args.model} with {args.val_language} doesn't exist!")
            raise NotImplementedError

        f = open(args.task + '_eval_log.txt', 'a')
        global_step, train_loss, best_val_metric, best_val_epoch, best_model_state_dict = total_canine_c_train(
            args=args,
            conll_dataset = train_dataset,
            kor_dataset  = validation_dataset,
            model=model,
            device=device,
            f=f
        )
        logging.info("global_step = %s, average training loss = %s", global_step, train_loss)
        logging.info("Best performance: Epoch=%d, Value=%s", best_val_epoch, best_val_metric)

        # Zero-shot evaluation
        model.load_state_dict(best_model_state_dict)
        model.eval()

        zero_shot_lang_list = ['korean', 'spanish', 'turkmen', 'maori', 'somali', 'uyghur', 'sinhala', 'quechua',
                               'assyrian', 'kashubian', 'ilocano', 'kyrgyz', 'kinyarwanda', 'esperanto', 'khmer',
                               'amharic', 'maltese', 'tajik', 'yoruba', 'marathi', 'javanese', 'urdu', 'malay',
                               'cebuano', 'croatian', 'malayalam', 'telugu', 'uzbek', 'punjabi']

        for lang in zero_shot_lang_list:
            if args.model == 'bio_canine_ch':
                logging.info(f'Loading BIO CANINE ch {lang} Dataset')
                zeroshot_dataset = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset(lang, 'validation', args.max_seq_len)
            elif args.model == 'bio_canine_epi':
                logging.info(f'Loading BIO CANINE epi {lang} Dataset')
                zeroshot_dataset = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset(lang, 'validation', args.max_seq_len)
            else:
                print(f"{lang} of {args.model} is not in the language list")
                raise NotImplementedError

            zeroshot_results, _ = total_canine_c_eval(
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

    ##### Eval #####
    # Evaluation on Train language's test data
    if args.do_predict:
        ##### Train Dataset Loading #####
        if args.model == 'bio_canine_ch' and args.train_language == 'english':  # for now, just use eng wikiann for training
            ##### BIO WikiAnn ch English Dataset #####
            logging.info('Loading BIO CANINE ch WikiAnn English Dataset')
            train_dataset = {}
            train_dataset['train'] = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset('english', 'train',
                                                                                           args.max_seq_len)
            train_dataset['validation'] = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset('english', 'validation',
                                                                                                args.max_seq_len)
            train_dataset['test'] = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset('english', 'test',
                                                                                          args.max_seq_len)
        elif args.model == 'bio_canine_epi' and args.train_language == 'english':  # for now, just use eng wikiann for training
            ##### BIO WikiAnn epi English Dataset #####
            logging.info('Loading BIO CANINE epi WikiAnn English Dataset')
            train_dataset = {}
            train_dataset['train'] = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset('english', 'train',
                                                                                             args.max_seq_len)
            train_dataset['validation'] = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset('english',
                                                                                                  'validation',
                                                                                                  args.max_seq_len)
            train_dataset['test'] = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset('english', 'test',
                                                                                            args.max_seq_len)
        else:
            print(f"{args.model} is not defined!")

        if args.model_ckpt_path == None:
            print("You should set model checkpoint path you want to evaluate")
            raise NotImplementedError
        else:
            # model ckpt load #
            model.load_state_dict(torch.load(args.model_ckpt_path))
            f = open(args.task + 'train_lan_test_log.txt', 'w')
            ##### Train Language's test set Evaluation #####
            results, _ = total_canine_c_eval(
                args=args,
                eval_dataset = train_dataset["test"],
                model=model,
                device=device
            )

            logging.info("\n***** Train Language Test results *****")
            f.write("***** Train Language Test results *****\n")
            for key in sorted(results.keys()):
                logging.info("\n  %s = %s", key, str(results[key]))
                f.write(str(key) + ': ' + str(results[key]) + '\n')
            f.close()

    ##### Only Zero-shot Eval #####
    if args.do_zero_predict:
        if args.model_ckpt_path == None:
            print("You should set model checkpoint path you want to evaluate")
            raise NotImplementedError
        else:
            logging.info(f'Only Zero-shot evaluation of {args.model} starts!')
            # model ckpt load #
            model.load_state_dict(torch.load(args.model_ckpt_path))
            model.eval()
            f = open(args.task + '_zero_eval_log.txt', 'w')
            ##### Zero-shot Language Validation set Evaluation #####
            zero_shot_lang_list = ['korean', 'spanish', 'turkmen', 'maori', 'somali', 'uyghur', 'sinhala', 'quechua',
                                   'assyrian', 'kashubian', 'ilocano', 'kyrgyz', 'kinyarwanda', 'esperanto', 'khmer',
                                   'amharic', 'maltese', 'tajik', 'yoruba', 'marathi', 'javanese', 'urdu', 'malay',
                                   'cebuano', 'croatian', 'malayalam', 'telugu', 'uzbek', 'punjabi',  'kurdish', 'sanskrit', 'interlingua', 'belarusian', 'oriya',
								                                  'guarani', 'sindhi']

            for lang in zero_shot_lang_list:
                if args.model == 'bio_canine_ch':
                    logging.info(f'Loading BIO CANINE ch {lang} Dataset')
                    zeroshot_dataset = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset(lang, 'validation',
                                                                                             args.max_seq_len)
                elif args.model == 'bio_canine_epi':
                    logging.info(f'Loading BIO CANINE epi {lang} Dataset')
                    zeroshot_dataset = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset(lang, 'validation',
                                                                                                 args.max_seq_len)
                else:
                    print(f"{lang} of {args.model} is not in the language list")
                    raise NotImplementedError

                zeroshot_results, _ = total_canine_c_eval(
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

    ##### Additional Zero-shot Eval #####
    if args.do_additional_predict:
        if args.model_ckpt_path == None:
            print("You should set model checkpoint path you want to evaluate")
            raise NotImplementedError
        else:
            logging.info(f'Only Zero-shot evaluation of {args.model} starts!')
            # model ckpt load #
            model.load_state_dict(torch.load(args.model_ckpt_path))
            model.eval()
            f = open(args.task + '_add_zero_eval_log.txt', 'w')
            ##### Zero-shot Language Validation set Evaluation #####
            zero_shot_lang_list = ['kyrgyz', 'oriya']

            for lang in zero_shot_lang_list:
                if args.model == 'bio_canine_ch':
                    logging.info(f'Loading BIO CANINE ch {lang} Dataset')
                    zeroshot_dataset = bio_canine_ch_whole_wikiann.BIOCanineChWikiAnnDataset(lang, 'validation',
                                                                                             args.max_seq_len)
                elif args.model == 'bio_canine_epi':
                    logging.info(f'Loading BIO CANINE epi {lang} Dataset')
                    zeroshot_dataset = bio_canine_epi_whole_wikiann.BIOCanineEpiWikiAnnDataset(lang, 'validation',
                                                                                                 args.max_seq_len)
                else:
                    print(f"{lang} of {args.model} is not in the language list")
                    raise NotImplementedError

                zeroshot_results, _ = total_canine_c_eval(
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
