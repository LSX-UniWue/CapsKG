# Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import random
import nlp_data_utils as data_utils
from nlp_data_utils import ABSATokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import math
from datasets import load_dataset

# domains = ['Video_Games','Toys_and_Games','Tools_and_Home_Improvement','Sports_and_Outdoors','Pet_Supplies',
#            'Patio_Lawn_and_Garden','Office_Products','Musical_Instruments','Movies_and_TV',
#            'Kindle_Store','Home_and_Kitchen','Health_and_Personal_Care','Grocery_and_Gourmet_Food','Electronics',
#            'Digital_Music','Clothing_Shoes_and_Jewelry','Cell_Phones_and_Accessories','CDs_and_Vinyl',
#            'Books','Beauty','Baby','Automotive','Apps_for_Android','Amazon_Instant_Video']


# domains = ['Antonym', 'AtLocation', 'DerivedFrom', 'EtymologicallyRelatedTo', 'FormOf',
#            'HasContext', 'IsA', 'RelatedTo' ,'Synonym',
#            'UsedFor'] # 10 tasks

domains = ['isLocatedIn', 'diedIn', 'wasBornIn', 'hasGender', 'happenedIn', 'isPoliticianOf', 'isCitizenOf', 'livesIn']

datasets = ['./dat/Yago_BERT_Base_Split/' + domain for domain in domains]


def get(logger=None, args=None):
    data = {}
    taskcla = []

    # Others
    # f_name = 'dsc_random_10'
    f_name = './data_prep/yago_' + str(args.ntasks)
    if not os.path.isfile(f_name): f_name = './data_prep/yago'
    # f_few_rel = load_dataset("few_rel")

    with open(f_name, 'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()
        # random.shuffle(random_sep)
    print('random_sep: ', random_sep)
    print('domains: ', domains)
    seq_inf = {}
    seq_inf['seq_file_name'] = f_name
    for t in range(args.ntasks):
        dataset = datasets[domains.index(random_sep[t])]
        rel_type = random_sep[t]  # TODO my: relation type is tracked here
        seq_inf[rel_type] = t
        data[t] = {}
        data[t]['name'] = dataset
        data[t]['ncla'] = 2  # TODO my it is number of classes

        print('dataset: ', dataset)
        logger.info(dataset)

        processor = data_utils.Yago_Processor()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        label_list = processor.get_labels(model_tokenizer=tokenizer)
        data[t]['ncla'] = len(label_list)

        train_examples, dict_all_train_labels = processor.get_train_examples(dataset, debug=args.debug,
                                                                             model_tokenizer=tokenizer,
                                                                             max_len=args.max_seq_length)

        if args.train_data_size > 0:  # TODO: for replicated results, better do outside (in prep_dsc.py), so that can save as a file
            random.Random(args.data_seed).shuffle(train_examples)  # more robust
            border = min(len(train_examples),args.train_data_size)
            train_examples = train_examples[:border]

        # print('##### FIRST 100', t)
        # first_100 = [x.text_a for x in train_examples[:100]]
        # last_100 = [x.text_a for x in train_examples[-100:]]
        # print('first 100 train samples:', first_100)
        # print('##### LAST 100', t)
        # print('last 100 train samples:', last_100)

        # TODO: to be conssitent, sometimes we want to convert it back and save
        # with open('./dat/dsc/'+random_sep[t]+'/train_'+str(args.train_data_size),'w') as fw:
        #     logger.info('train_examples: '+str(len(train_examples)))
        #     for train_example in train_examples:
        #         fw.writelines(train_example.text_a + '\t' + train_example.label + '\n')
        #     logger.info('train_example.text_a: '+str(train_example.text_a))

        # Doble Saved:  So that no change on data even if we change the seed
        # if args.train_data_size > 0: # in some case, I might not cutting anything
        #     #So that no change on data even if we change the seed
        #     if not os.path.exists("./dat/dsc/"+str(random_sep[t])+"/train_data_"+str(args.train_data_size)+'_'+str(args.data_seed)):
        #         torch.save(train_examples,"./dat/dsc/"+str(random_sep[t])+"/train_data_"+str(args.train_data_size)+'_'+str(args.data_seed))
        #         logger.info("  save")
        #     else: #in that case, change seed will not change data
        #         train_examples = torch.load("./dat/dsc/"+str(random_sep[t])+"/train_data_"+str(args.train_data_size)+'_'+str(0)) #always use 0
        #         logger.info("  load")

        num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs
        # num_train_steps = int(len(train_examples) / args.train_batch_size) * args.num_train_epochs

        train_features = data_utils.convert_examples_to_features_knowledge(
            train_examples, label_list, args.max_seq_length, tokenizer)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks)

        data[t]['train'] = train_data
        data[t]['num_train_steps'] = num_train_steps

        valid_examples, dict_all_train_dev_labels = processor.get_dev_examples(dataset, debug=args.debug,
                                                                               model_tokenizer=tokenizer,
                                                                               labels_dict=dict_all_train_labels,
                                                                               max_len=args.max_seq_length)
        if '_SW' in rel_type:
            for r_key in dict_all_train_dev_labels.keys():
                for e1_key in dict_all_train_dev_labels[r_key].keys():
                    dict_all_train_dev_labels[r_key][e1_key] = []
        # No need to change valid for DSC
        # if args.dev_data_size > 0:
        #     random.Random(args.data_seed).shuffle(valid_examples) #more robust
        #     valid_examples = valid_examples[:args.dev_data_size]

        valid_features = data_utils.convert_examples_to_features_knowledge(
            valid_examples, label_list, args.max_seq_length, tokenizer)

        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask,
                                   valid_all_label_ids, valid_all_tasks)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        data[t]['valid'] = valid_data



        processor = data_utils.Yago_Processor()
        label_list = processor.get_labels(model_tokenizer=tokenizer)
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        eval_examples = processor.get_test_examples(dataset, debug=args.debug, model_tokenizer=tokenizer,
                                                    labels_dict=dict_all_train_dev_labels,
                                                    max_len=args.max_seq_length)

        common_vocab = set(
            open('./dat/Yago_BERT_Base_Split/graph_vocab.txt', "r").read().splitlines())
        vocab_vector, _ = data_utils.build_model_vocab_vector(tokenizer, common_vocab)
        eval_features = data_utils.convert_examples_to_features_knowledge(eval_examples, label_list,
                                                                               args.max_seq_length, tokenizer,
                                                                               subset='test',
                                                                               vocab_vector=vocab_vector)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)
        all_possible_labels = torch.stack([f.all_possible_labels for f in eval_features])
        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids, all_tasks,
                                  all_possible_labels)
        # Run prediction for full data

        data[t]['test'] = eval_data


        taskcla.append((t, int(data[t]['ncla'])))

    # Others
    n = 0
    for t in data.keys():
        n += data[t]['ncla']
    data['ncla'] = n
    data['task_seq_inf'] = seq_inf  # ToDo my for random order and fast load

    return data, taskcla
