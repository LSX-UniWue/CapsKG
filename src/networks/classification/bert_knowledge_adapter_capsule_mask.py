#coding: utf-8
import sys
import torch
from transformers import BertModel, BertConfig
# from transformers.modeling_bert import BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertForMaskedLM
import utils
from torch import nn

sys.path.append("./networks/base/")
from my_transformers import MyBertModel
from nlp_data_utils import ABSATokenizer

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()
        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        args.build_adapter_capsule_mask = True
        self.bert = MyBertModel.from_pretrained(args.bert_model,config=config,args=args)
        self.tokenizer = ABSATokenizer.from_pretrained(args.bert_model)

        for param in self.bert.parameters():
            # param.requires_grad = True
            param.requires_grad = False


        if args.apply_bert_output and args.apply_bert_attention_output:
            adapter_masks = \
                [self.bert.encoder.layer[layer_id].attention.output.adapter_capsule_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].output.adapter_capsule_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif args.apply_bert_output:
            adapter_masks = \
                [self.bert.encoder.layer[layer_id].output.adapter_capsule_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif args.apply_bert_attention_output:
            adapter_masks = \
                [self.bert.encoder.layer[layer_id].attention.output.adapter_capsule_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.bert.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        for adapter_mask in adapter_masks:
            for param in adapter_mask.parameters():
                param.requires_grad = True

        self.taskcla=taskcla
        self.last=torch.nn.ModuleList()
        # init MLMHead, HF Work around
        original_model_for_mask_pred = BertForMaskedLM.from_pretrained(args.bert_model)
        # freeze and reuse classification layer
        kg_maskhead_layer = BertOnlyMLMHead(config)
        kg_maskhead_layer.load_state_dict(original_model_for_mask_pred.cls.state_dict())
        for param in kg_maskhead_layer.parameters():
            param.requires_grad = False
        for t,n in self.taskcla:
            self.last.append(kg_maskhead_layer)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args
        self.config = config
        self.num_task = len(taskcla)
        self.num_kernel = 3

        print('BERT KNOWLEDGE ADAPTER CAPSULE MASK')

        return

    def forward(self,t,input_ids, segment_ids, input_mask, targets, s=1):


        output_dict = \
            self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                      targets=None,t=t,s=s)

        sequence_output, pooled_output = output_dict['outputs']

        y=[]
        for t_x, i in self.taskcla:
            y.append(self.last[t_x](sequence_output)) # TODO my code masking tasl, input is sequence

        masks = self.mask(t,s)

        output_dict['y'] = y
        output_dict['masks'] = masks

        return output_dict

    def mask(self,t,s):
        masks = {}
        for layer_id in range(self.config.num_hidden_layers):

            if self.args.apply_bert_output:
                fc1_key = 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1' #gfc1
                fc2_key = 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2' #gfc2

                masks[fc1_key],masks[fc2_key] = self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.mask(t,s)

                key = 'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger' #gfc1
                masks[key] = self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.mask(t,s)

            if self.args.apply_bert_attention_output:
                fc1_key = 'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1' #gfc1
                fc2_key = 'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2' #gfc2

                masks[fc1_key],masks[fc2_key] = self.bert.encoder.layer[layer_id].attention.output.adapter_capsule_mask.mask(t,s)

                key = 'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger' #gfc1
                masks[key] = self.bert.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.mask(t,s)

        return masks




    def get_view_for(self,n,p,masks):
        for layer_id in range(self.config.num_hidden_layers):
            if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1.weight':
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1.bias':
                return masks[n.replace('.bias','')].data.view(-1)
            elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2.weight':
                post=masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                pre=masks[n.replace('.weight','').replace('fc2','fc1')].data.view(1,-1).expand_as(p)
                return torch.min(post,pre)
            elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2.bias':
                return masks[n.replace('.bias','')].data.view(-1)

            elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1.weight':
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1.bias':
                return masks[n.replace('.bias','')].data.view(-1)
            elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2.weight':
                post=masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                pre=masks[n.replace('.weight','').replace('fc2','fc1')].data.view(1,-1).expand_as(p)
                return torch.min(post,pre)
            elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2.bias':
                return masks[n.replace('.bias','')].data.view(-1)

            if n == \
            'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger.weight': #gfc1
                # print('tsv_capsules not none')
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n == \
            'bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger.bias': #gfc1
                return masks[n.replace('.bias','')].data.view(-1)

            if n == \
            'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger.weight': #gfc1
                # print('tsv_capsules not none')
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n == \
            'bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger.bias': #gfc1
                return masks[n.replace('.bias','')].data.view(-1)

        return None

    def get_view_for_tsv(self,n,t):
        for layer_id in range(self.config.num_hidden_layers):
            if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.route_weights':
                # print('not none')
                return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t].data.view(1,-1,1,1)
            for c_t in range(self.num_task):
                if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias':
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight':
                    # print('not none')
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias':
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.bias':
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                for m_t in range(self.num_kernel):
                    if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.bias':
                        # print('not none')
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

            if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.route_weights':
                # print('not none')
                return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t].data.view(1,-1,1,1)

            for c_t in range(self.num_task):
                if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias':
                    return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight':
                    # print('not none')
                    return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias':
                    return self.bert.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.bias':
                    return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data


                for m_t in range(self.num_kernel):
                    if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='bert.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.bert.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

        return 1 #if no condition is satified