#coding: utf-8
import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F
sys.path.append("./networks/base/")
from my_transformers import MyBertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertForMaskedLM
from nlp_data_utils import ABSATokenizer

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()
        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        args.build_adapter = True
        self.bert = MyBertModel.from_pretrained(args.bert_model,config=config,args=args)
        self.tokenizer = ABSATokenizer.from_pretrained(args.bert_model)

        #BERT fixed all ===========
        for param in self.bert.parameters():
            # param.requires_grad = True
            param.requires_grad = False

        #But adapter is open

        #Only adapters are trainable

        if args.apply_bert_output and args.apply_bert_attention_output:
            adaters = \
                [self.bert.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif args.apply_bert_output:
            adaters = \
                [self.bert.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif self.apply_bert_attention_output:
            adaters = \
                [self.bert.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)]


        for adapter in adaters:
            for param in adapter.parameters():
                param.requires_grad = True

        self.taskcla=taskcla
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # init MLMHead, HF Work around
        original_model_for_mask_pred = BertForMaskedLM.from_pretrained(args.bert_model)
        kg_maskhead_layer = BertOnlyMLMHead(config)
        kg_maskhead_layer.load_state_dict(original_model_for_mask_pred.cls.state_dict())
        for param in kg_maskhead_layer.parameters():
            param.requires_grad = False
        self.args = args
        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(args.bert_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(kg_maskhead_layer)


        print('BERT ADAPTER')

        return

    def forward(self,input_ids, segment_ids, input_mask):
        output_dict = {}


        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        pooled_output = self.dropout(pooled_output)
        #shared head

        if 'dil' in self.args.scenario:
            y = self.last(pooled_output)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](sequence_output))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)

        return output_dict