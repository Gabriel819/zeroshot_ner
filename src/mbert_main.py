import os
import argparse
import torch
import datetime
import logging

from utils.misc import set_seed
from dataset import bio_mbert_whole
from utils.mbert_evaluation import mbert_eval
from utils.mbert_training import mbert_train
from model.char_ner_mbert import CharNERBertModel
from transformers import BertTokenizer, BertModel

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
        choices=['bio_mbert']
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
        default=5e-5, type=float, help="The initial learning rate for Adam.")
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
        "--do_additional_zero_predict",
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
    if args.model == 'bio_mbert':
        ##### Model - BIO Tagging mBERT #####
        logging.info('Loading BIO Tagging mBERT model')
        tmp_model = BertModel.from_pretrained("google-bert/bert-base-multilingual-cased")
        model_config = tmp_model.config
        model = CharNERBertModel(model_config, 7)
    else:
        print("Wrong Model input")
        raise NotImplementedError
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # whole model param#
    print('Whole model number of params:', n_parameters)

    ##### Train Dataset Loading #####
    # if args.model == 'bio_mbert' and args.train_language == 'english': # for now, just use eng wikiann for training
    #     ##### BIO Tagging mBERT WikiAnn English Dataset #####
    #     logging.info('Loading BIO Tagging mBERT WikiAnn English Dataset')
    #     train_dataset = {}
    #     train_dataset['train'] = bio_mbert_whole.BIOMbertWholeDataset('english', 'train', args.max_seq_len)
    #     train_dataset['validation'] = bio_mbert_whole.BIOMbertWholeDataset('english', 'validation', args.max_seq_len)
    #     train_dataset['test'] = bio_mbert_whole.BIOMbertWholeDataset('english', 'test', args.max_seq_len)
    # else:
    #     print(f"Train language {args.train_language} is not implemented!")
    #     raise NotImplementedError

    ##### Zero-shot Test Dataset Loading #####
    ########## BIO tagging(7 classes) mBERT ##########
    # 1. Korean
    # logging.info('Loading BIO mBERT Korean Dataset')
    # korean_dataset = bio_mbert_whole.BIOMbertWholeDataset('korean', 'validation', args.max_seq_len)
    # # 2. Spanish
    # logging.info('Loading BIO mBERT Spanish Dataset')
    # spanish_dataset = bio_mbert_whole.BIOMbertWholeDataset('spanish', 'validation', args.max_seq_len)
    # # 3. Sinhala
    # logging.info('Loading BIO mBERT Sinhala Dataset')
    # sinhala_dataset = bio_mbert_whole.BIOMbertWholeDataset('sinhala', 'validation', args.max_seq_len)
    # # 4. Maori
    # logging.info('Loading BIO mBERT Maori Dataset')
    # maori_dataset = bio_mbert_whole.BIOMbertWholeDataset('maori', 'validation', args.max_seq_len)
    # # 5. Kashubian
    # logging.info('Loading BIO mBERT Kashubian Dataset')
    # kashubian_dataset = bio_mbert_whole.BIOMbertWholeDataset('kashubian', 'validation', args.max_seq_len)
    # # 6. Somali
    # logging.info('Loading BIO mBERT Somali Dataset')
    # somali_dataset = bio_mbert_whole.BIOMbertWholeDataset('somali', 'validation', args.max_seq_len)
    # # 7. Quechua
    # logging.info('Loading BIO mBERT Quechua Dataset')
    # quechua_dataset = bio_mbert_whole.BIOMbertWholeDataset('quechua', 'validation', args.max_seq_len)
    # # 8. Uyghur
    # logging.info('Loading BIO mBERT Uyghur Dataset')
    # uyghur_dataset = bio_mbert_whole.BIOMbertWholeDataset('uyghur', 'validation', args.max_seq_len)
    # # 9. Assyrian
    # logging.info('Loading BIO mBERT Assyrian Dataset')
    # assyrian_dataset = bio_mbert_whole.BIOMbertWholeDataset('assyrian', 'validation', args.max_seq_len)
    # # 10. Kinyarwanda
    # logging.info('Loading BIO mBERT Kinyarwanda Dataset')
    # kinyarwanda_dataset = bio_mbert_whole.BIOMbertWholeDataset('kinyarwanda', 'validation', args.max_seq_len)
    # # 11. Kyrgyz
    # logging.info('Loading BIO mBERT Kyrgyz Dataset')
    # kyrgyz_dataset = bio_mbert_whole.BIOMbertWholeDataset('kyrgyz', 'validation', args.max_seq_len)
    # # 12. Ilocano
    # logging.info('Loading BIO mBERT Ilocano Dataset')
    # ilocano_dataset = bio_mbert_whole.BIOMbertWholeDataset('ilocano', 'validation', args.max_seq_len)
    # # 13. Esperanto
    # logging.info('Loading BIO mBERT Esperanto Dataset')
    # esperanto_dataset = bio_mbert_whole.BIOMbertWholeDataset('esperanto', 'validation', args.max_seq_len)
    # # 14. Khmer
    # logging.info('Loading BIO mBERT Khmer Dataset')
    # khmer_dataset = bio_mbert_whole.BIOMbertWholeDataset('khmer', 'validation', args.max_seq_len)
    # # 15. Turkmen
    # logging.info('Loading BIO mBERT Turkmen Dataset')
    # turkmen_dataset = bio_mbert_whole.BIOMbertWholeDataset('turkmen', 'validation', args.max_seq_len)
    # # 16. Amharic
    # logging.info('Loading BIO mBERT Amharic Dataset')
    # amharic_dataset = bio_mbert_whole.BIOMbertWholeDataset('amharic', 'validation', args.max_seq_len)
    # # 17. Maltese
    # logging.info('Loading BIO mBERT Maltese Dataset')
    # maltese_dataset = bio_mbert_whole.BIOMbertWholeDataset('maltese', 'validation', args.max_seq_len)
    # # 18. Tajik
    # logging.info('Loading BIO mBERT Tajik Dataset')
    # tajik_dataset = bio_mbert_whole.BIOMbertWholeDataset('tajik', 'validation', args.max_seq_len)
    # # 17. Yoruba
    # logging.info('Loading BIO mBERT Yoruba Dataset')
    # yoruba_dataset = bio_mbert_whole.BIOMbertWholeDataset('yoruba', 'validation', args.max_seq_len)
    # # 18. Marathi
    # logging.info('Loading BIO mBERT Marathi Dataset')
    # marathi_dataset = bio_mbert_whole.BIOMbertWholeDataset('marathi', 'validation', args.max_seq_len)
    # # 19. Javanese
    # logging.info('Loading BIO mBERT Javanese Dataset')
    # javanese_dataset = bio_mbert_whole.BIOMbertWholeDataset('javanese', 'validation', args.max_seq_len)
    # # 20. Urdu
    # logging.info('Loading BIO mBERT Urdu Dataset')
    # urdu_dataset = bio_mbert_whole.BIOMbertWholeDataset('urdu', 'validation', args.max_seq_len)
    # # 21. Malay
    # logging.info('Loading BIO mBERT Malay Dataset')
    # malay_dataset = bio_mbert_whole.BIOMbertWholeDataset('malay', 'validation', args.max_seq_len)
    # # 22. Cebuano
    # logging.info('Loading BIO mBERT Cebuano Dataset')
    # cebuano_dataset = bio_mbert_whole.BIOMbertWholeDataset('cebuano', 'validation', args.max_seq_len)
    # # 23. Croatian
    # logging.info('Loading BIO mBERT Croatian Dataset')
    # croatian_dataset = bio_mbert_whole.BIOMbertWholeDataset('croatian', 'validation', args.max_seq_len)
    # # 24. Malayalam
    # logging.info('Loading BIO mBERT Malayalam Dataset')
    # malayalam_dataset = bio_mbert_whole.BIOMbertWholeDataset('malayalam', 'validation', args.max_seq_len)
    # # 25. Telugu
    # logging.info('Loading BIO mBERT Telugu Dataset')
    # telugu_dataset = bio_mbert_whole.BIOMbertWholeDataset('telugu', 'validation', args.max_seq_len)
    # # 26. Uzbek
    # logging.info('Loading BIO mBERT Uzbek Dataset')
    # uzbek_dataset = bio_mbert_whole.BIOMbertWholeDataset('uzbek', 'validation', args.max_seq_len)
    # # 27. Punjabi
    # logging.info('Loading BIO mBERT Punjabi Dataset')
    # punjabi_dataset = bio_mbert_whole.BIOMbertWholeDataset('punjabi', 'validation', args.max_seq_len)

    ##### Train #####
    # Log args
    logging.info('Using the following arguments for training:')
    for k, v in vars(args).items():
        logging.info("* %s: %s", k, v)
    if args.do_train:
        ##### Train Dataset Loading #####
        if args.model == 'bio_mbert' and args.train_language == 'english':  # for now, just use eng wikiann for training
            ##### BIO Tagging mBERT WikiAnn English Dataset #####
            logging.info('Loading BIO Tagging mBERT WikiAnn English Dataset')
            train_dataset = {}
            train_dataset['train'] = bio_mbert_whole.BIOMbertWholeDataset('english', 'train', args.max_seq_len)
            train_dataset['validation'] = bio_mbert_whole.BIOMbertWholeDataset('english', 'validation',
                                                                               args.max_seq_len)
            train_dataset['test'] = bio_mbert_whole.BIOMbertWholeDataset('english', 'test', args.max_seq_len)
        else:
            print(f"Train language {args.train_language} is not implemented!")
            raise NotImplementedError

        ##### Zero-shot Test Dataset Loading #####
        ########## BIO tagging(7 classes) mBERT ##########
        # 1. Korean
        logging.info('Loading BIO mBERT Korean Dataset')
        korean_dataset = bio_mbert_whole.BIOMbertWholeDataset('korean', 'validation', args.max_seq_len)

        f = open(args.task + '_eval_log.txt', 'w')

        global_step, train_loss, best_val_metric, best_val_epoch, best_model_state_dict = total_mbert_train(
            args=args,
            eng_dataset = train_dataset,
            kor_dataset  = korean_dataset,
            model=model,
            device=device,
            f=f
        )
        logging.info("global_step = %s, average training loss = %s", global_step, train_loss)
        logging.info("Best performance: Epoch=%d, Value=%s", best_val_epoch, best_val_metric)

        # Zero-shot evaluation
        model.load_state_dict(best_model_state_dict)
        model.eval()

        zero_shot_lang_list = ['korean', 'spanish', 'turkmen', 'maori', 'somali', 'uyghur', 'sinhala', 'quechua', 'assyrian',
                               'kashubian', 'ilocano', 'kyrgyz', 'kinyarwanda', 'esperanto', 'khmer', 'amharic', 'maltese',
                               'tajik', 'yoruba', 'marathi', 'javanese', 'urdu', 'malay', 'cebuano', 'croatian', 'malayalam',
                               'telugu', 'uzbek', 'punjabi', 'kurdish', 'sanskrit', 'interlingua', 'belarusian', 'oriya',
                               'guarani', 'sindhi']

        for lang in zero_shot_lang_list:
            logging.info(f'Loading BIO mBERT {lang} Dataset')
            zeroshot_dataset = bio_mbert_whole.BIOMbertWholeDataset(lang, 'validation', args.max_seq_len)

            zeroshot_results, _ = mbert_eval(
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
        if args.model == 'bio_mbert' and args.train_language == 'english':  # for now, just use eng wikiann for training
            ##### BIO Tagging mBERT WikiAnn English Dataset #####
            logging.info('Loading BIO Tagging mBERT WikiAnn English Dataset')
            train_dataset = {}
            train_dataset['train'] = bio_mbert_whole.BIOMbertWholeDataset('english', 'train', args.max_seq_len)
            train_dataset['validation'] = bio_mbert_whole.BIOMbertWholeDataset('english', 'validation',
                                                                               args.max_seq_len)
            train_dataset['test'] = bio_mbert_whole.BIOMbertWholeDataset('english', 'test', args.max_seq_len)
        else:
            print(f"Train language {args.train_language} is not implemented!")
            raise NotImplementedError

        if args.model_ckpt_path == None:
            print("You should set model checkpoint path you want to evaluate")
            raise NotImplementedError
        else:
            # model ckpt load #
            model.load_state_dict(torch.load(args.model_ckpt_path))
            f = open(args.task + '_train_lan_test_log.txt', 'w')
            ##### Train Language's test set Evaluation #####
            results, _ = mbert_eval(
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
                                   'cebuano', 'croatian', 'malayalam', 'telugu', 'uzbek', 'punjabi', 'kurdish', 'sanskrit', 'interlingua', 'belarusian', 'oriya',
                               'guarani', 'sindhi']

            for lang in zero_shot_lang_list:
                logging.info(f'Loading BIO mBERT {lang} Dataset')
                zeroshot_dataset = bio_mbert_whole.BIOMbertWholeDataset(lang, 'validation', args.max_seq_len)

                zeroshot_results, _ = mbert_eval(
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
    if args.do_additional_zero_predict:
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
			# zero_shot_lang_list = ['kyrgyz', 'oriya']
            zero_shot_lang_list = ['sindhi']
            for lang in zero_shot_lang_list:
                logging.info(f'Loading BIO mBERT {lang} Dataset')
                zeroshot_dataset = bio_mbert_whole.BIOMbertWholeDataset(lang, 'validation', args.max_seq_len)

                zeroshot_results, _ = mbert_eval(
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
