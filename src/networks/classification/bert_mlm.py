#coding: utf-8
import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertForMaskedLM
from nlp_data_utils import ABSATokenizer

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()
        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)
        self.tokenizer = ABSATokenizer.from_pretrained(args.bert_model)

        '''
        In case you want to fix some layers
        '''
        #BERT fixed some ===========
        # modules = [self.bert.embeddings, self.bert.encoder.layer[:args.activate_layer_num]] #Replace activate_layer_num by what you want
        # modules = [self.bert.encoder.layer[-1]]
        #
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False

        #BERT fixed all ===========
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        self.taskcla=taskcla
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # init MLMHead, HF Work around
        original_model_for_mask_pred = BertForMaskedLM.from_pretrained(args.bert_model)

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(args.bert_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            # for t,n in self.taskcla: # TODO my code add mask prediction head for each KG
            #     kg_maskhead_layer = BertOnlyMLMHead(config)
            #     # init MLMHead, HF Work around
            #     ## for example, I want my model's last encoder layer's weight = original model's last encoder's weight
            #     kg_maskhead_layer.load_state_dict(original_model_for_mask_pred.cls.state_dict())
            #     self.last.append(kg_maskhead_layer)
            kg_maskhead_layer = BertOnlyMLMHead(config)
            # init MLMHead, HF Work around
            ## for example, I want my model's last encoder layer's weight = original model's last encoder's weight
            kg_maskhead_layer.load_state_dict(original_model_for_mask_pred.cls.state_dict())
            self.last.append(kg_maskhead_layer)


        print('DIL BERT')

        return

    def forward(self,input_ids, segment_ids, input_mask):
        output_dict = {}

        sequence_output, pooled_output = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        pooled_output = self.dropout(pooled_output)

        if 'dil' in self.args.scenario:
            y = self.last(pooled_output)
        elif 'til' in self.args.scenario:
            y=[]
            y.append(self.last[0](sequence_output)) #TODO my only one task per model

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)

        return output_dict