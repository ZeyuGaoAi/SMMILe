import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from random import random, randint

"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""

class SQMIL(nn.Module):
    def __init__(self, gate=True, size_arg = "small", dropout = False, subtyping=False, n_classes=2, n_refs=1, fea_dim=1024, 
                 instance_loss_fn=nn.CrossEntropyLoss(reduction='none')):
        nn.Module.__init__(self)
        
        self.size_dict = {"small": [fea_dim, 8, 4], "big": [fea_dim, 12, 6]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        
        self.det_net = nn.Sequential(*fc)
        
        # self.det_net = nn.Linear(size[2], n_classes)# nn.Sequential(*fc)
        self.cls_net = nn.Linear(size[1], n_classes)
    
        ref_net = [nn.Linear(size[1], n_classes+1) for i in range(n_refs)]
        self.ref_net = nn.ModuleList(ref_net)  

        self.n_classes = n_classes
        self.n_refs = n_refs
        self.instance_loss_fn=instance_loss_fn

        initialize_weights(self)
        
    @staticmethod
    def create_targets(length, cls, device):
        return torch.full((length, ), cls, device=device).long()
        
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cls_net = self.cls_net.to(device)
        self.det_net = self.det_net.to(device)
        self.ref_net = self.ref_net.to(device)
        
    def find_candidate(self, final_score, h, label, tp_rate=0.01, np_rate=0.01):
        device = h.device
        
        all_targets = []
        all_instances = []
        all_weights = []
        np_index_set = []
        for cls in range(self.n_classes):
            cls_final_score = final_score[:,cls]
            cls_final_score = (cls_final_score-cls_final_score.min())/(cls_final_score.max()-cls_final_score.min())
            if cls in label:
                # top 10% & score > 0.5
                tp_score, tp_index = torch.topk(cls_final_score, max(1,int(len(cls_final_score)*tp_rate))) # int(len(cls_final_score)*tp_rate)
                tp_index = tp_index[tp_score>0.5]
                tp_weights = tp_score[tp_score>0.5]
                # todo find their neigbors??
                tp_h = torch.index_select(h, dim=0, index=tp_index)
                tp_target = self.create_targets(len(tp_index), cls, device)
                all_instances.append(tp_h)
                all_targets.append(tp_target)
                all_weights.append(tp_weights)
                
        # down in all classes 10% & score < 0.5
        final_score_mean = torch.mean(final_score, dim=1)
        np_score, np_index = torch.topk(-final_score_mean, int(len(final_score_mean)*np_rate))
        np_score = -np_score
        np_index = np_index[np_score<0.5]
        np_weights = 1 - np_score[np_score<0.5]
        np_h = torch.index_select(h, dim=0, index=np_index)
        np_target = self.create_targets(len(np_index), self.n_classes, device)
        all_instances.append(np_h)
        all_targets.append(np_target)
        all_weights.append(np_weights)

        all_targets = torch.cat(all_targets, 0)
        all_instances = torch.cat(all_instances, 0)
        all_weights = torch.cat(all_weights, 0)
            
        return all_targets, all_instances, all_weights
    
    def find_candidate_ref(self, final_score, h, label, rate=0.01):
        device = h.device
        
        all_targets = []
        all_instances = []
        all_weights = []
        for cls in range(final_score.shape[-1]):
            cls_final_score = final_score[:,cls]
            if cls in label or cls == self.n_classes: # only for contained classes and background class
                # top 10% & score > 0.5
                tp_score, tp_index = torch.topk(cls_final_score, int(len(cls_final_score)*rate))
                tp_index = tp_index[tp_score>0.5]
                tp_weights = tp_score[tp_score>0.5]
                # todo find their neigbors
                tp_h = torch.index_select(h, dim=0, index=tp_index)
                tp_target = self.create_targets(len(tp_index), cls, device)
                all_instances.append(tp_h)
                all_targets.append(tp_target)
                all_weights.append(tp_weights)
        
        all_targets = torch.cat(all_targets, 0)
        all_instances = torch.cat(all_instances, 0)
        all_weights = torch.cat(all_weights, 0)
            
        return all_targets, all_instances, all_weights
    
    def forward2(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, epsilon=1e-10):
        # logits-based 
        
        device = h.device
        
        det_logit, h= self.det_net(h) # N x cls
        
        cls_logit = self.cls_net(h) # N x cls

        cls_score = F.softmax(cls_logit, dim=1) # the cls prob of each patch
        det_score = F.softmax(det_logit, dim=0) # the det prob (attention) of each patch

        final_logits = cls_logit * det_score # N x cls

        final_logits = torch.sum(final_logits, dim=0)

        final_score = F.softmax(final_logits, dim=0)
        
        Y_prob = final_score
        
        Y_hat = torch.topk(Y_prob, 1, dim = 0)[1]
        
        results_dict = {}
        
        if return_features:
            results_dict.update({'cls_logits': cls_logit})
            results_dict.update({'cls_scores': cls_score})
            results_dict.update({'det_logits': det_logit})
            results_dict.update({'det_scores': det_score})
            
        return final_score, Y_prob, Y_hat, results_dict

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, epsilon=1e-10):
        # score based
        
        device = h.device
        
        det_logit, h= self.det_net(h) # N x cls
        
        cls_logit = self.cls_net(h) # N x cls

        cls_score = F.softmax(cls_logit, dim=1) # the cls prob of each patch
        det_score = F.softmax(det_logit, dim=0) # the det prob (attention) of each patch
        
        final_score = cls_score * det_score # N x cls
        
        Y_prob = torch.clamp(torch.sum(final_score, dim=0), min=epsilon, max=1-epsilon) # 1x3
        # Y_prob = torch.sum(final_score, dim=0)
        
        Y_hat = torch.topk(Y_prob, 1, dim = 0)[1]
        
        results_dict = {}
        
        if return_features:
            results_dict.update({'cls_logits': cls_logit})
            results_dict.update({'cls_scores': cls_score})
            results_dict.update({'det_logits': det_logit})
            results_dict.update({'det_scores': det_score})
            
        return final_score, Y_prob, Y_hat, results_dict
    
class SQMIL_NIC(SQMIL):
    def __init__(self, gate=True, size_arg = "small", dropout = False,
                 n_classes=2, n_refs=3, drop_rate=0.1, fea_dim=1024,
                 instance_loss_fn=nn.CrossEntropyLoss(reduction='none')):
        nn.Module.__init__(self)
        
        self.size_dict = {"small": [fea_dim, 128, 64], "big": [fea_dim, 512, 256]} # 512 256 overfitting
        size = self.size_dict[size_arg]
        
        if size_arg == 'small':
            conv_nic = [nn.Conv2d(size[0], size[1], kernel_size=3, stride=1, padding=1, bias=False)] #  dilation=1
        else:
            conv_nic = [nn.Conv2d(size[0], size[1], kernel_size=1, stride=1, padding=0, bias=False)]
            
        conv_nic.append(nn.BatchNorm2d(size[1]))
        conv_nic.append(nn.ReLU())
        if dropout:
            conv_nic.append(nn.Dropout(drop_rate)) # dropout little better than no dropout & dropout2d 
        self.conv_nic = nn.Sequential(*conv_nic)
        
        self.det_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        self.cls_net = nn.Linear(size[1], n_classes)

        ref_net = [nn.Linear(size[1], n_classes+1) for i in range(n_refs)]
        self.ref_net = nn.ModuleList(ref_net)

        self.n_classes = n_classes
        self.n_refs = n_refs
        self.instance_loss_fn=instance_loss_fn

        initialize_weights(self)
        
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_nic = self.conv_nic.to(device)
        self.cls_net = self.cls_net.to(device)
        self.det_net = self.det_net.to(device)
        self.ref_net = self.ref_net.to(device)
        
    def drop_with_score_old(self, final_score, drop_rate=0.1, label=None):
        # weighted drop based on final score, by drop_rate => 0.1 is better than 0.25 
        device = final_score.device
        tensor_rd = torch.rand(final_score.shape).to(device)
        tensor_mask = torch.ones(final_score.shape[0]).to(device)
        final_score_rd = torch.pow(tensor_rd, 1/final_score)
        _, indices = torch.topk(final_score_rd, int(final_score.shape[0]*drop_rate), dim=0)
        if label is None:
            drop_mask = [tensor_mask.index_fill(0, indices[:,i], 0) for i in range(indices.shape[1])]
            drop_mask = torch.stack(drop_mask, dim=0).T
        else:
            drop_mask = torch.ones(final_score.shape).to(device)
            for i in range(indices.shape[1]):
                if i in label:
                    drop_mask = drop_mask.index_fill(0, indices[:,i], 0)
        return drop_mask
    
    def drop_with_score(self, final_score, label=None):
        
        # no drop_rate is necessary, drop instances base on their scores.
        device = final_score.device
        tensor_rd = torch.rand(final_score.shape).to(device)
        drop_mask = torch.ones(final_score.shape).to(device)
        
        final_score_norm = torch.stack([(final_score[:,i]-final_score[:,i].min())/(final_score[:,i].max()-final_score[:,i].min()) 
                                        for i in range(final_score.shape[-1])]).T
        drop_index = tensor_rd>final_score_norm
        
        # only drop the column of label that wsi belongs
        for i in range(self.n_classes):
            if i in label:
                drop_mask[:,i] = drop_index.int()[:,i]
        
        return drop_mask
    
    def find_candidate_sp(self, sp_score, sp, all_sp, h, label, tp_rate=0.01, np_rate=0.01):
        device = h.device
        
        all_targets = []
        all_instances = []
        all_weights = []
        for cls in range(self.n_classes):
            if cls in label:
                cls_sp_score = sp_score[:,cls]
                # pos
                tp_score, tp_index = torch.topk(cls_sp_score, max(1,int(len(cls_sp_score)*tp_rate)))
                tp_h = []
                for tpi in tp_index:
                    tp_hi = h[sp==all_sp[tpi]]
                    tp_h.append(tp_hi)
                tp_h = torch.cat(tp_h, dim=0)
                tp_targets = self.create_targets(tp_h.shape[0], cls, device)
                tp_weights = torch.stack([tp_score for i in range(tp_h.shape[0])])
                all_instances.append(tp_h)
                all_targets.append(tp_targets)
                all_weights.append(tp_weights)
        # neg
        sp_score_mean = torch.mean(sp_score, dim=1)
        np_score, np_index = torch.topk(-sp_score_mean, max(1,int(len(cls_sp_score)*np_rate))) 
        np_h = []
        for npi in np_index:
            np_hi = h[sp==all_sp[tpi]]
            np_h.append(np_hi)
        np_h = torch.cat(np_h, dim=0)
        np_targets = self.create_targets(np_h.shape[0], self.n_classes, device)
        np_weights = torch.stack([(1+np_score) for i in range(np_h.shape[0])])
        all_instances.append(np_h)
        all_targets.append(np_targets)
        all_weights.append(np_weights)
        
        all_targets = torch.cat(all_targets, 0)
        all_instances = torch.cat(all_instances, 0)
        all_weights = torch.cat(all_weights, 0)
        return all_targets, all_instances, all_weights
        
        
    def superpixel_sample(self, all_sp, sp, det_logit, cls_logit, g_num):
        det_logit_sampled = [[] for i in range(g_num)]
        cls_logit_sampled = [[] for i in range(g_num)]
        final_score_sampled = []
        for sp_index in all_sp:
            det_logit_sub = det_logit[sp==sp_index]
            cls_logit_sub = cls_logit[sp==sp_index]
            for i in range(g_num):
                rd = randint(0, det_logit_sub.shape[0]-1)
                det_logit_sampled[i].append(det_logit_sub[rd])
                cls_logit_sampled[i].append(cls_logit_sub[rd])
        for i in range(g_num):
            det_temp = torch.stack(det_logit_sampled[i])
            cls_temp = torch.stack(cls_logit_sampled[i])
            
            cls_score_sampled = F.softmax(cls_temp, dim=1) # the cls prob of each patch
            det_score_sampled = F.softmax(det_temp, dim=0) # the det prob (attention) of each patch
            
            final_score_sampled.append(cls_score_sampled * det_score_sampled) # N x cls
            
        return final_score_sampled

    def _del_edges(self, loss):
        device = loss.device
        tensor_rd = torch.rand(loss.shape).to(device)
        drop_mask = torch.ones(loss.shape).to(device)
        valid_loss = loss[~torch.isnan(loss) & ~torch.isinf(loss)]

        if valid_loss.numel() > 0:
            loss_norm = (valid_loss - valid_loss.min()) / (valid_loss.max() - valid_loss.min())
        else:
            loss_norm = torch.zeros_like(loss)
        drop_index = tensor_rd>loss_norm
        return drop_index
    
    def markov_field_energy(self, scores, sp_indexs, sp_list, adjacency_matrix, unary_weight=0.8, pairwise_weight=0.2, 
                            drop=True, loss_fn=nn.MSELoss(reduction='none')):
    
        unary_energy = 0
        pairwise_energy = 0
        
        # calculate the score of each sp, mean or max?
        sp_scores = torch.zeros((sp_list.max()+1, scores.shape[-1]))
        
        if unary_weight != 0:

            for sp_index in sp_list:
                one_sp_scores = scores[sp_indexs==sp_index]
                one_sp_scores_mean = torch.mean(one_sp_scores, dim=0)
                sp_scores[sp_index] = one_sp_scores_mean
                # constrain the similarity within each superpixel
                unary_loss = torch.mean(loss_fn(one_sp_scores, one_sp_scores_mean.repeat(one_sp_scores.shape[0],1)), dim=-1)
                # randomly del some edges
                if drop:
                    unary_mask = self._del_edges(unary_loss)
                    unary_energy += torch.mean(unary_loss * unary_mask)
                else:
                    unary_energy += torch.mean(unary_loss)
        
        
        if pairwise_weight!= 0:

            for sp_index in sp_list:
                if sp_index == 0: # skip outliers in sp
                    continue
                one_adj = adjacency_matrix[sp_index,:]
                one_sp_score = sp_scores[sp_index]
                adj_sp_scores = sp_scores[(one_adj==1)] # the arrangment of adj and sp_list are same
                # constrain the similarity between superpixel neighbors
                pairwise_loss = torch.mean(loss_fn(adj_sp_scores, one_sp_score.repeat(adj_sp_scores.shape[0],1)), dim=-1)
                # randomly del some edges
                if drop:
                    pairwise_mask = self._del_edges(pairwise_loss)
                    pairwise_energy += torch.mean(pairwise_loss * pairwise_mask)
                else:
                    pairwise_energy += torch.mean(pairwise_loss)

        # weighted sum
        energy_loss = unary_weight * unary_energy + pairwise_weight * pairwise_energy

        return energy_loss

    def forward(self, h, mask, sp, adj, label=None, instance_eval=False, inst_rate=0.01, 
                return_features=False, group_numbers=1, superpixels=False, drop_with_score=False, 
                mrf = False, s = False, tau=1, epsilon=1e-10):
        
        mrf_loss = 0
        instance_loss = 0

        device = h.device
        
        f_h, f_w = np.where(mask==1)
        
        sp = sp[f_h,f_w]
        sp_list = np.unique(sp)
        
        
        h_raw = h[:,f_h,f_w].T

        h = self.conv_nic(h.unsqueeze(0)).squeeze(0)
        
        h = h[:,f_h,f_w].T

        det_logit, _ = self.det_net(h) # N x cls
        
        cls_logit = self.cls_net(h) # N x cls
        
        cls_score = F.softmax(cls_logit, dim=1) # the cls prob of each patch
        
        det_score = F.softmax(det_logit, dim=0) # the det prob (attention) of each patch
        
        final_score = cls_score * det_score # N x cls
        
        Y_prob = [torch.clamp(torch.sum(final_score, dim=0), min=epsilon, max=1-epsilon)] # basic prob is necessary to consider
        
        final_score_sp = final_score.clone().detach()
        
        Y_hat = torch.topk(Y_prob[0], 1, dim = 0)[1]

        if drop_with_score:

            drop_mask = self.drop_with_score(final_score, label) # N * cls
            final_score_droped = final_score * drop_mask
            Y_prob_drop = torch.clamp(torch.sum(final_score_droped, dim=0), min=epsilon, max=1-epsilon) # 1x3

            Y_prob.append(Y_prob_drop)
            
        if superpixels:
        
            # use superpixel to sample
            all_sp_score = []
            sp_score_list = self.superpixel_sample(sp_list, sp, det_logit, cls_logit, group_numbers)
    
            for sp_score in sp_score_list:
                Y_prob_sp = torch.clamp(torch.sum(sp_score, dim=0), min=epsilon, max=1-epsilon) # 1x3
                all_sp_score.append(sp_score)
                Y_prob.append(Y_prob_sp)

                if drop_with_score:
                    drop_mask = self.drop_with_score(sp_score, label) # N * cls
                    sp_score_droped = sp_score * drop_mask
                    Y_prob_sp = torch.clamp(torch.sum(sp_score_droped, dim=0), min=epsilon, max=1-epsilon) # 3
                    Y_prob.append(Y_prob_sp)

            Y_hat = torch.topk(torch.mean(torch.stack(Y_prob), dim=0), 1, dim = 0)[1]

            # max pooling for each superpixel
            sp_score_max = torch.max(torch.stack(all_sp_score), dim=0)[0]
            sp_score_mean = torch.mean(torch.stack(all_sp_score), dim=0)
            
            # aggregation for sp
            for sp_index in range(len(sp_list)):
                final_score_sp[sp==sp_list[sp_index]] = sp_score_mean[sp_index]
            
        ref_score = final_score
        
        if instance_eval:
            ref_logits = []
            for r in range(self.n_refs):
                if r == 0:
                    cand_targets, cand_insts, cand_weights = self.find_candidate(final_score_sp, h, label, 
                                                                   tp_rate=inst_rate, np_rate=inst_rate)
                else:
                    cand_targets, cand_insts, cand_weights = self.find_candidate_ref(ref_score, h, label, rate=inst_rate)
                if len(cand_targets)!=0:
                    cand_inst_logits = self.ref_net[r](cand_insts)
                    instance_loss += torch.mean(self.instance_loss_fn(cand_inst_logits, cand_targets)) # *cand_weights
                ref_logits = self.ref_net[r](h) # n_ref : N x (cls+1)
                ref_score = F.softmax(ref_logits, dim=1) # N x (cls+1)

                # MRF energy constraint for final_score
                if mrf:
                    mrf_loss += tau * self.markov_field_energy(ref_score, sp, sp_list, adj)/self.n_refs

        results_dict = {}
        results_dict = {'instance_loss': instance_loss}
        results_dict.update({'mrf_loss': mrf_loss})
        
        if return_features:
            results_dict.update({'cls_logits': cls_logit})
            results_dict.update({'cls_scores': cls_score})
            results_dict.update({'det_logits': det_logit})
            results_dict.update({'det_scores': det_score})
            results_dict.update({'h_raw': h_raw})

        return final_score_sp, Y_prob, Y_hat, ref_score, results_dict

class SQMIL_NIC_SINGLE(SQMIL_NIC):
    def __init__(self, gate=True, size_arg = "small", dropout = False,
                 n_classes=1, n_refs=3, drop_rate=0.1, fea_dim=1024,
                 instance_loss_fn=nn.CrossEntropyLoss(reduction='none')):
        nn.Module.__init__(self)
        
        self.size_dict = {"small": [fea_dim, 512, 256], "big": [fea_dim, 384, 256]} #
        size = self.size_dict[size_arg]
        
        conv_nic = [nn.Conv2d(size[0], size[1], kernel_size=1, stride=1, padding=1, bias=False)] #  1x1 for camelyon
        conv_nic.append(nn.BatchNorm2d(size[1]))
        conv_nic.append(nn.ReLU())
        if dropout:
            conv_nic.append(nn.Dropout(drop_rate)) 
        self.conv_nic = nn.Sequential(*conv_nic)
        
        self.det_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1) # det score is 1 for normal vs tumor
        
        self.cls_net = nn.Linear(size[1], 1)
        
        ref_net = [nn.Linear(size[1], 2) for i in range(n_refs)]
        self.ref_net = nn.ModuleList(ref_net)  

        self.n_classes = n_classes
        self.n_refs = n_refs
        self.instance_loss_fn=instance_loss_fn

        initialize_weights(self)
            
    def drop_with_score_single(self, final_score, label=None):
        
        # no drop_rate is necessary, drop instances base on their scores.
        device = final_score.device
        tensor_rd = torch.rand(final_score.shape).to(device)
        drop_mask = torch.ones(final_score.shape).to(device)
        
        final_score_norm = torch.stack([(final_score[:,i]-final_score[:,i].min())/(final_score[:,i].max()-final_score[:,i].min()) 
                                        for i in range(final_score.shape[-1])]).T
        
        drop_index = tensor_rd>final_score_norm
        
        # only drop the column of label that wsi belongs
        # if 1 in label: # positive: drop high score
        drop_mask[:,0] = drop_index.int()[:,0]
        
        return drop_mask
    
    def consistency_penalty(self, final_score, loss_fn=nn.MSELoss()):
        # for negative sample, all instances should have similar scores.
        # it is useful but not necessary
        sp_score_mean = torch.mean(final_score, dim=0)
        consis_loss = loss_fn(final_score, sp_score_mean.repeat(final_score.shape[0],1))
        
        return consis_loss

    def find_candidate_ref_single(self, final_score, h, label, rate=0.01):
        device = h.device
        
        all_targets = []
        all_instances = []
        all_weights = []
        
        if 1 in label:
            for cls in range(2):
                cls_final_score = final_score[:,cls]
                # top 10% & score > 0.5
                tp_score, tp_index = torch.topk(cls_final_score, int(len(cls_final_score)*rate))
                tp_index = tp_index[tp_score>0.5]
                tp_weights = tp_score[tp_score>0.5]
                # todo find their neigbors
                tp_h = torch.index_select(h, dim=0, index=tp_index)
                tp_target = self.create_targets(len(tp_index), cls, device)
                
                all_instances.append(tp_h)
                all_targets.append(tp_target)
                all_weights.append(tp_weights)
                
        else:
            cls_final_score = final_score[:,1]
            tp_score, tp_index = torch.topk(cls_final_score, int(len(cls_final_score)*rate))
            tp_weights = tp_score
            tp_h = torch.index_select(h, dim=0, index=tp_index)
            tp_target = self.create_targets(len(tp_index), 0, device)
            
            all_instances.append(tp_h)
            all_targets.append(tp_target)
            all_weights.append(tp_weights)
                

        
        all_targets = torch.cat(all_targets, 0)
        all_instances = torch.cat(all_instances, 0)
        all_weights = torch.cat(all_weights, 0)
            
        return all_targets, all_instances, all_weights    
    
    def find_candidate_single(self, final_score, h, label, tp_rate=0.01, np_rate=0.01):
        device = h.device
        
        all_targets = []
        all_instances = []
        all_weights = []
        np_index_set = []
        
        cls_final_score = final_score[:, 0]
        cls_final_score = (cls_final_score-cls_final_score.min())/(cls_final_score.max()-cls_final_score.min())
        
        if 1 in label:
            tp_score, tp_index = torch.topk(cls_final_score, max(1,int(len(cls_final_score)*tp_rate)))
            
            tp_index = tp_index[tp_score>0.5]
            tp_weights = tp_score[tp_score>0.5]
            
            tp_h = torch.index_select(h, dim=0, index=tp_index)
            tp_target = self.create_targets(len(tp_index), 1, device)
            all_instances.append(tp_h)
            all_targets.append(tp_target)
            all_weights.append(tp_weights)
            
            np_score, np_index = torch.topk(-cls_final_score, max(1,int(len(cls_final_score)*np_rate)))
            np_score = -np_score
            np_index = np_index[np_score<0.5]
            np_weights = np_score[np_score<0.5]
            
            np_h = torch.index_select(h, dim=0, index=np_index)
            np_target = self.create_targets(len(np_index), 0, device)
            all_instances.append(np_h)
            all_targets.append(np_target)
            all_weights.append(np_weights)

        else:
            np_score, np_index = torch.topk(cls_final_score, max(1,int(len(cls_final_score)*np_rate)))
            
            np_h = torch.index_select(h, dim=0, index=np_index)
            np_target = self.create_targets(len(np_index), 0, device)
            all_instances.append(np_h)
            all_targets.append(np_target)
            all_weights.append(np_score)      

        all_targets = torch.cat(all_targets, 0)
        all_instances = torch.cat(all_instances, 0)
        all_weights = torch.cat(all_weights, 0)
            
        return all_targets, all_instances, all_weights
        
    def superpixel_sample_single(self, all_sp, sp, det_logit, cls_logit, g_num):
        det_logit_sampled = [[] for i in range(g_num)]
        cls_logit_sampled = [[] for i in range(g_num)]
        final_score_sampled = []
        for sp_index in all_sp:
            det_logit_sub = det_logit[sp==sp_index]
            cls_logit_sub = cls_logit[sp==sp_index]
            for i in range(g_num):
                rd = randint(0, det_logit_sub.shape[0]-1)
                det_logit_sampled[i].append(det_logit_sub[rd])
                cls_logit_sampled[i].append(cls_logit_sub[rd])
        for i in range(g_num):
            det_temp = torch.stack(det_logit_sampled[i])
            cls_temp = torch.stack(cls_logit_sampled[i])
            
            cls_score_sampled = torch.sigmoid(cls_temp) # the cls prob of each patch
            det_score_sampled = F.softmax(det_temp, dim=0) # the det prob (attention) of each patch
            
            final_score_sampled.append(cls_score_sampled * det_score_sampled) # N x cls
            
        return final_score_sampled
    
    def pred_score(self, h, mask):

        device = h.device
        f_h, f_w = np.where(mask==1)

        h = self.conv_nic(h.unsqueeze(0)).squeeze(0)
        
        h = h[:,f_h,f_w].T
        
        det_logit, _ = self.det_net(h) # N x 1
        cls_logit = self.cls_net(h) # N x 1

        cls_score = torch.sigmoid(cls_logit) # the cls prob of each patch

        det_score = F.softmax(det_logit, dim=0) # the det prob (attention) of each patch

        final_score = cls_score * det_score # N x 1
        
        final_score = final_score.clone().detach() 
        
        return final_score

    def forward(self, h, mask, sp, adj, label=None, instance_eval=False, inst_rate=0.01, 
                return_features=False, group_numbers=1, superpixels=False, drop_with_score=False, 
                mrf = False, consistency = False, tau=1, epsilon=1e-10):
        

        mrf_loss = 0
        instance_loss = 0
        consist_loss = 0

        device = h.device
        f_h, f_w = np.where(mask==1)
        
        sp = sp[f_h,f_w]
       
        sp_list = np.unique(sp)

        h = self.conv_nic(h.unsqueeze(0)).squeeze(0)
        
        h = h[:,f_h,f_w].T
        
        det_logit, _ = self.det_net(h) # N x 1
        cls_logit = self.cls_net(h) # N x 1

        cls_score = torch.sigmoid(cls_logit) # the cls prob of each patch

        det_score = F.softmax(det_logit, dim=0) # the det prob (attention) of each patch

        final_score = cls_score * det_score # N x 1

        Y_prob_all = torch.clamp(torch.sum(final_score, dim=0), min=epsilon, max=1-epsilon) # 1

        Y_prob = [Y_prob_all]

        Y_hat = (Y_prob_all>0.5).cuda()*1
        
        # for drops & superpixels
        
        final_score_sp = final_score.clone().detach()
        
        if consistency and (0 in label):
            
            consist_loss += self.consistency_penalty(det_logit)
        
        # if not superpixels and drop_with_score:
        # for single class classification, only positive WSI need to be drop
        else:

            if drop_with_score:

                drop_mask = self.drop_with_score_single(final_score, label) # N * cls

                final_score_droped = final_score * drop_mask

                Y_prob_drop = torch.clamp(torch.sum(final_score_droped, dim=0), min=epsilon, max=1-epsilon) # 1

                Y_prob.append(Y_prob_drop)

            if superpixels:

                # use superpixel to sample

                all_sp_score = []

                sp_score_list = self.superpixel_sample_single(sp_list, sp, det_logit, cls_logit, group_numbers)

                for sp_score in sp_score_list:

                    Y_prob_sp = torch.clamp(torch.sum(sp_score, dim=0), min=epsilon, max=1-epsilon) # 1x3

                    all_sp_score.append(sp_score)
                    
                    Y_prob.append(Y_prob_sp)

                    if drop_with_score and (Y_prob_sp>0.5):
                        drop_mask = self.drop_with_score_single(sp_score, label) # N * cls
                        sp_score_droped = sp_score * drop_mask
                        Y_prob_sp = torch.clamp(torch.sum(sp_score_droped, dim=0), min=epsilon, max=1-epsilon) # 3
                        Y_prob.append(Y_prob_sp)

                # use original Y_hat is better

                # max pooling for each superpixel
                sp_score_max = torch.max(torch.stack(all_sp_score), dim=0)[0]
                sp_score_mean = torch.mean(torch.stack(all_sp_score), dim=0)
            
            
        ref_score = final_score_sp

        if instance_eval:
            ref_logits = [] 
            for r in range(self.n_refs): # self.n_refs
                if r == 0:
                    cand_targets, cand_insts, cand_weights = self.find_candidate_single(final_score_sp, h, label, 
                                                                  tp_rate=inst_rate, np_rate=inst_rate)
                else:
                    cand_targets, cand_insts, cand_weights = self.find_candidate_ref_single(ref_score, h, label, rate=inst_rate)
                if len(cand_targets)!=0:
                    cand_inst_logits = self.ref_net[r](cand_insts)
                    instance_loss += torch.mean(self.instance_loss_fn(cand_inst_logits, cand_targets)) # *cand_weights
                ref_logits = self.ref_net[r](h) # n_ref : N x (cls+1)
                ref_score = F.softmax(ref_logits, dim=1) # N x (cls+1)
                
                # MRF energy constraint for final_score 0.1 better for camelyon
                if mrf:
                    mrf_loss += tau * self.markov_field_energy(ref_score, sp, sp_list, adj)/self.n_refs
            
        results_dict = {}
        results_dict = {'instance_loss': instance_loss}
        results_dict.update({'mrf_loss': mrf_loss})
        results_dict.update({'consist_loss': consist_loss})
        
        if return_features:
            results_dict.update({'features': h})
        
        return final_score_sp, Y_prob, Y_hat, ref_score, results_dict
