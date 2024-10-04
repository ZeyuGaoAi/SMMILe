import numpy as np
import torch
from utils.utils import *
from torch.optim import lr_scheduler
import os
import pandas as pd
from datasets.dataset_nic import save_splits
from models.model_smmile import SMMILe
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.bi_tempered_loss_pytorch import bi_tempered_binary_logistic_loss

def cal_pos_acc(pos_label,pos_score,inst_rate):
    df_score = pd.DataFrame([pos_label,pos_score]).T
    df_score = df_score.sort_values(by=1)
    df_score_top = df_score.iloc[-int(df_score.shape[0]*inst_rate):,:]
    total_num_pos = df_score_top.shape[0]
    if total_num_pos:
        return df_score_top[0].sum()/total_num_pos
    else:
        return float('nan')

def cal_neg_acc(neg_label, neg_score, inst_rate):
    df_score = pd.DataFrame([neg_label,neg_score]).T
    df_score = df_score.sort_values(by=1)
    df_score_down = df_score.iloc[:int(df_score.shape[0]*inst_rate),:]
    total_num_neg = df_score_down.shape[0]
    if total_num_neg:
        return df_score_down[0].sum()/total_num_neg
    else:
        return float('nan')

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        # Y_hat输出的是值的形式而非向量，Y也要值的形式
        Y = np.atleast_1d(Y.squeeze().cpu().numpy())
        Y_hat = Y_hat.squeeze().cpu().numpy()
        for true in Y:
            self.data[true]["count"] += 1
            if true in Y_hat:
                self.data[true]["correct"] += 1
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    if args.reverse_train_val:
        val_split, train_split, test_split = datasets
    else:
        train_split, val_split, test_split = datasets

    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    
    args.bi_loss = False
    loss_fn = nn.functional.binary_cross_entropy
    
    if args.bag_loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()
        
    elif args.bag_loss == 'bibce':
        args.bi_loss = True
        
    print('Done!')
    
    print('\nInit Model...', end=' ')
    
    model_dict = {'dropout': args.drop_out, 'drop_rate': args.drop_rate,
                  'multi_label': args.multi_label, 'n_classes': args.n_classes, 
                  'fea_dim': args.fea_dim, "size_arg": args.model_size,
                  'n_refs': args.n_refs}

    model = SMMILe(**model_dict)

    
    if args.models_dir is not None:
        ckpt_path = os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(cur))
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt, strict=False)
            print('\nThe model has been loaded from %s' % ckpt_path)
        else:
            print('\nThe model will train from scrash')
    
    model.relocate()
    print_network(model)
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=100, verbose = True)
    else:
        early_stopping = None
    print('Done!')
    
    
    if args.ref_start_epoch == 0 and args.inst_refinement: # continue training the model from ckpt 
        ref_start = True
    else:
        ref_start = False
    

    for epoch in range(args.max_epochs):
        if args.model_type in ['smmile']:
            train_loop_smmile(epoch, model, train_loader, optimizer, writer, loss_fn, ref_start, args)
            stop = validate_smmile(cur, epoch, model, val_loader, early_stopping, writer, loss_fn, 
                                    ref_start, scheduler, args)
        else:
            raise NotImplementedError 
        
        if (stop and not ref_start and args.inst_refinement) or (epoch == args.ref_start_epoch and args.inst_refinement):
            ref_start = True
            early_stopping = EarlyStopping(patience = 50, stop_epoch=100, verbose = True)
        elif stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint_best.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, val_iauc, _= summary(model, val_loader, args)
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, test_error, test_auc, test_iauc, acc_logger = summary(model, test_loader, args)
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/val_iauc', val_iauc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.add_scalar('final/test_iauc', test_iauc, 0)
        writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error , test_iauc, val_iauc
        
def train_loop_smmile(epoch, model, loader, optimizer, writer = None, loss_fn = None, ref_start = False, args=None):
    
    n_classes = args.n_classes
    consistency = args.consistency
    bi_loss = args.bi_loss
    multi_label = args.multi_label
    drop_with_score = args.drop_with_score
    D = args.D
    superpixel = args.superpixel
    G = args.G
    inst_refinement = args.inst_refinement
    inst_rate = args.inst_rate
    mrf = args.mrf
    tau = args.tau
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.
    inst_loss = 0.
    m_loss = 0.
    cons_loss = 0.
    
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []
    
    if not ref_start:
        inst_refinement = False
    
    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        if multi_label:
            label = label.to(device)
            index_label = torch.nonzero(label.squeeze()).to(device)
            
        else:
            index_label = label.to(device)
            label = torch.zeros(n_classes)
            label[index_label.long()] = 1
            label = label.to(device)
        
        total_loss = 0
        total_loss_value = 0
        inst_loss_value = 0
        mrf_loss_value = 0
        consist_loss_value = 0
        
        data = data.to(device)
        mask = cors[1]
        sp = cors[2]
        adj = cors[3]
        
        score, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp, adj, index_label, 
                                                              group_numbers = G,
                                                              superpixels=superpixel, 
                                                              drop_with_score=drop_with_score,
                                                              drop_times = D,
                                                              instance_eval=inst_refinement,
                                                              inst_rate=inst_rate,
                                                              mrf=mrf,
                                                              tau=tau,
                                                              consistency = consistency)
        
        if inst_label!=[] and not all(x == -1 for x in inst_label):
            all_inst_label += inst_label
            label_hot = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
            inst_probs_tmp = [] 
            inst_preds_tmp = []
            # if not inst_refinement:
            inst_score = score.detach().cpu().numpy() # Nxcls 不含正常

            # max-min standard all_inst_score and all_inst_score_pos
            for class_idx in range(n_classes):
                
                if class_idx not in index_label:
                    inst_score[:,class_idx] = [0]*len(inst_label)
                    continue
                    
                inst_score_one_class = inst_score[:,class_idx]

                inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                                            max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))

                inst_probs_tmp.append(inst_score_one_class)    
                    
                inst_score[:,class_idx] = inst_score_one_class
                
                inst_probs[class_idx]+=inst_score_one_class
                
                inst_preds[class_idx]+=[0 if i<0.5 else 1 for i in inst_score_one_class]
                inst_preds_tmp.append([0 if i<0.5 else 1 for i in inst_score_one_class])
                
                inst_binary_labels[class_idx]+=list(label_hot[:,class_idx])

                pos_accs_each_wsi[class_idx].append(cal_pos_acc(label_hot[:,class_idx],inst_score_one_class,inst_rate))
                
            if inst_preds_tmp != []:
                
                inst_preds_tmp = np.mean(np.stack(inst_preds_tmp), axis=0) # Nx1
                
            else:
                inst_preds_tmp = [0]*len(inst_label)
                
            inst_preds_neg = [1 if i==0 else 0 for i in inst_preds_tmp]
            inst_preds[n_classes] += inst_preds_neg
            inst_binary_labels[n_classes]+=list(label_hot[:,n_classes])
            
            if inst_probs_tmp:
                neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #n类平均，越低越是neg

                neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))

            else:
                neg_score = [0]*len(inst_label)
            neg_accs_each_wsi.append(cal_neg_acc(label_hot[:,n_classes], neg_score, inst_rate))
                
            if inst_refinement:
                inst_score = ref_score.detach().cpu().numpy() #有正常类
                # pos_score = score.detach().cpu().numpy() #final_score_sp 没有正常类

                # # max-min standard all_inst_score and all_inst_score_pos
                # for class_idx in range(n_classes):
                #     if class_idx not in index_label:
                #         continue
                #     inst_score_one_class = pos_score[:,class_idx]

                #     inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                #                                 max(inst_score_one_class.max()-inst_score_one_class.min(), 1e-10))
                    
                #     inst_probs_tmp.append(inst_score_one_class)
                    # pos_accs_each_wsi[class_idx].append(cal_pos_acc(label_hot[:,class_idx],inst_score_one_class,inst_rate))


                # if inst_probs_tmp:
                #     neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #n类平均，越低越是neg

                #     neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))

                # else:
                #     neg_score = [0]*len(inst_label)
                # neg_accs_each_wsi.append(cal_neg_acc(label_hot[:,n_classes], neg_score, inst_rate))


            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score

        acc_logger.log(Y_hat, index_label)
        
        loss = loss_fn(Y_prob[0], label.squeeze().float())/len(Y_prob)

        for one_prob in Y_prob[1:]:
            if bi_loss:
                loss += bi_tempered_binary_logistic_loss(one_prob, label, 0.2, 1., reduction='mean')/len(Y_prob)
            else:
                loss += loss_fn(one_prob, label.squeeze().float())/len(Y_prob)
        

        loss_value = loss.item()

        total_loss += loss
        total_loss_value += loss_value

        if inst_refinement:
            instance_loss = results_dict['instance_loss']
            if instance_loss!=0:
                total_loss += instance_loss 
                inst_loss_value += instance_loss.item()

        if mrf:
            mrf_loss = results_dict['mrf_loss'] 
            if mrf_loss!=0:
                total_loss += mrf_loss 
                mrf_loss_value += mrf_loss.item()
        
        consist_loss = results_dict['consist_loss']
        if consist_loss!=0:
            total_loss += 1e-1*consist_loss 
            consist_loss_value += consist_loss.item()


        error = calculate_error(Y_hat, label)
        train_error += error
        
        train_loss += total_loss_value
        inst_loss += inst_loss_value
        m_loss += mrf_loss_value
        cons_loss += consist_loss_value

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    inst_loss /= len(loader)
    m_loss /= len(loader)
    cons_loss /= len(loader)
    
    # calculate inst_auc and inst_acc   
    
    all_inst_score = np.concatenate(all_inst_score,axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg] #转化为是normal类的概率 越高越好
    
    # get inst_pred inst_acc
    if not inst_refinement:
        inst_accs = []
        for i in range(n_classes+1):
            if len(inst_binary_labels[i])==0:
                continue
            inst_accs.append(accuracy_score(inst_binary_labels[i], inst_preds[i]))
            print('class {}'.format(str(i)))
            print(classification_report(inst_binary_labels[i], inst_preds[i],zero_division=1))
        inst_acc = np.mean(inst_accs)
    else:
        all_inst_pred = np.argmax(all_inst_score,axis=1)
        all_inst_pred = list(all_inst_pred)
  
        inst_acc = accuracy_score(all_inst_label, all_inst_pred)
        print(classification_report(all_inst_label, all_inst_pred,zero_division=1))

    # get inst_auc
    inst_aucs = []
    if not inst_refinement:
        for class_idx in range(n_classes):
            inst_score_sub = inst_probs[class_idx]
            
            if len(inst_score_sub)==0:
                continue
            fpr, tpr, _ = roc_curve(inst_binary_labels[class_idx], inst_score_sub)
            inst_aucs.append(calc_auc(fpr, tpr))
            
        fpr,tpr,_ = roc_curve(inst_binary_labels[n_classes], all_normal_score)
        inst_aucs.append(calc_auc(fpr, tpr))
    else:
        binary_inst_label = label_binarize(all_inst_label, classes=[i for i in range(n_classes+1)])
        for class_idx in range(n_classes+1):
            if class_idx in all_inst_label:
                inst_score_sub = all_inst_score[:,class_idx]
                if len(inst_score_sub)==0 or np.isnan(inst_score_sub).any():
                    continue
                fpr, tpr, _ = roc_curve(binary_inst_label[:, class_idx], inst_score_sub)
                inst_aucs.append(calc_auc(fpr, tpr))
            else:
                inst_aucs.append(float('nan'))
    inst_auc = np.nanmean(np.array(inst_aucs))
    
    # calculate pos acc and neg acc
    pos_accs = [sum(i)/len(i) if len(i) else 0 for i in pos_accs_each_wsi]
    pos_acc = np.nanmean(np.array(pos_accs))
    pos_acc_str = "seleted pos %f acc: "%(inst_rate)
    for i in range(n_classes):
        if i<n_classes-1:
            pos_acc_str+='class'+ str(i)+' '+str(pos_accs[i])+', '
        else:
            pos_acc_str+='class'+ str(i)+' '+str(pos_accs[i])
    print(pos_acc_str)
    neg_acc = sum(neg_accs_each_wsi)/len(neg_accs_each_wsi)
    print("seleted neg %f acc: %f" % (inst_rate, neg_acc))


    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, mrf_loss: {:.4f},cons_loss: {:.4f}, inst_loss: {:.4f}, inst_acc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc, m_loss, cons_loss, inst_loss, inst_acc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if acc is None:
            acc = 0
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        for i in range(n_classes):
            writer.add_scalar('train/pos_acc_{}'.format(str(i)), pos_accs[i], epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)

def validate_smmile(cur, epoch, model, loader, early_stopping = None, writer = None, loss_fn = None, 
                     ref_start=False, scheduler = None, args = None):

    n_classes = args.n_classes
    multi_label = args.multi_label
    superpixel = args.superpixel
    G = args.G
    inst_refinement = args.inst_refinement
    inst_rate = args.inst_rate
    bi_loss = args.bi_loss
    results_dir = args.results_dir
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_pos = []
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros((len(loader), n_classes))
    
    if not ref_start:
        inst_refinement = False
    
    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            if multi_label:
                label = label.to(device)
                index_label = torch.nonzero(label.squeeze()).to(device)
            else:
                index_label = label.to(device)
                label = torch.zeros(n_classes)
                label[index_label.long()] = 1
                label = label.to(device)
            
            
            data = data.to(device)
            mask = cors[1]
            sp = cors[2]
            adj = cors[3]
            score, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp,adj, index_label, 
                                                                  group_numbers=G,
                                                                  superpixels = superpixel,
                                                                  instance_eval=inst_refinement,
                                                                  inst_rate=inst_rate)

            if inst_label!=[] and not all(x == -1 for x in inst_label):
                all_inst_label += inst_label
                label_hot = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
                inst_probs_tmp = [] 
                inst_preds_tmp = []
                if not inst_refinement:
                    inst_score = score.detach().cpu().numpy()

                    # max-min standard all_inst_score and all_inst_score_pos
                    for class_idx in range(n_classes):
                        # if class_idx not in Y_hat:
                        if class_idx not in index_label:
                            inst_score[:,class_idx] = [0]*len(inst_label)
                            continue

                        inst_score_one_class = inst_score[:,class_idx]
                        # if len(set(inst_score_one_class))>1:
                        inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())
                                                    /max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))
                        inst_score_one_class = list(inst_score_one_class)
                        inst_score[:,class_idx] = inst_score_one_class

                        inst_probs[class_idx]+=inst_score_one_class
                        inst_probs_tmp.append(inst_score_one_class)                  

                        inst_preds[class_idx]+=[0 if i<0.5 else 1 for i in inst_score_one_class]
                        inst_preds_tmp.append([0 if i<0.5 else 1 for i in inst_score_one_class])

                        inst_binary_labels[class_idx]+=list(label_hot[:,class_idx])

                        pos_accs_each_wsi[class_idx].append(cal_pos_acc(label_hot[:,class_idx],
                                                                        inst_score_one_class,inst_rate))

                    if inst_preds_tmp !=[]:
                        inst_preds_tmp = np.mean(np.stack(inst_preds_tmp), axis=0)
                    else:
                        inst_preds_tmp = [0]*len(inst_label)
                    inst_preds_tmp = [1 if i==0 else 0 for i in inst_preds_tmp]
                    inst_preds[n_classes] += inst_preds_tmp
                    inst_binary_labels[n_classes]+=list(label_hot[:,n_classes])

                    if inst_probs_tmp:
                        neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #三类平均，越低越是neg
#                         if len(set(neg_score))>1:
                        neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
#                         neg_score = list(neg_score)
                    else:
                        neg_score = [0]*len(inst_label)
                    neg_accs_each_wsi.append(cal_neg_acc(label_hot[:,n_classes],neg_score,inst_rate))



                else:
                    inst_score = ref_score.detach().cpu().numpy()
                    pos_score = score.detach().cpu().numpy() #final_score_sp

                    # max-min standard all_inst_score and all_inst_score_pos
                    for class_idx in range(n_classes):
                        if class_idx not in index_label:
                            continue
                        inst_score_one_class = pos_score[:,class_idx]
#                         if len(set(inst_score_one_class))>1:
                        inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())
                                                    /max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))

                        inst_probs_tmp.append(inst_score_one_class)
                        pos_accs_each_wsi[class_idx].append(cal_pos_acc(label_hot[:,class_idx],
                                                                        inst_score_one_class,inst_rate))

                    if inst_probs_tmp:
                        neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #三类平均，越低越是neg
#                         if len(set(neg_score))>1:
                        neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
#                         neg_score = list(neg_score)
                    else:
                        neg_score = [0]*len(inst_label)
                    neg_accs_each_wsi.append(cal_neg_acc(label_hot[:,n_classes],neg_score,inst_rate))


                all_inst_score.append(inst_score)
                all_inst_score_neg += neg_score
            
            acc_logger.log(Y_hat, index_label)
            
            loss = loss_fn(Y_prob[0], label.squeeze().float())/len(Y_prob)

            for one_prob in Y_prob[1:]:
                if bi_loss:
                    loss += bi_tempered_binary_logistic_loss(one_prob, label, 0.2, 1., reduction='mean')/len(Y_prob)
                else:
                    loss += loss_fn(one_prob, label.squeeze().float())/len(Y_prob)
                    
            Y_prob = Y_prob[0]
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.squeeze().detach().cpu().numpy()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            
            if inst_refinement:
                instance_loss = results_dict['instance_loss']
                if instance_loss!=0:
                    val_loss += instance_loss.item()
            

    val_error /= len(loader)
    val_loss /= len(loader)
    
    # calculate inst_auc and inst_acc   
    all_inst_score = np.concatenate(all_inst_score,axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg] #转化为是normal类的概率 越高越好
    
    # get inst_pred inst_acc
    if not inst_refinement:
        inst_accs = []
        inst_p_macros = []
        inst_r_macros = []
        inst_f1_macros = []
        for i in range(n_classes+1):
            if len(inst_binary_labels[i])==0:
                continue
            inst_accs.append(accuracy_score(inst_binary_labels[i], inst_preds[i]))
            print('class {}'.format(str(i)))
            print(classification_report(inst_binary_labels[i], inst_preds[i],zero_division=1))

            inst_p_macros.append(precision_score(inst_binary_labels[i], inst_preds[i],zero_division=1))
            inst_r_macros.append(recall_score(inst_binary_labels[i], inst_preds[i],zero_division=1))
            inst_f1_macros.append(f1_score(inst_binary_labels[i], inst_preds[i],zero_division=1))
            
        inst_acc = np.mean(inst_accs)
        inst_p_macro = np.mean(inst_p_macros)
        inst_r_macro = np.mean(inst_r_macros)
        inst_f1_macro = np.mean(inst_f1_macros)
            
    else:
        all_inst_pred = np.argmax(all_inst_score,axis=1)
        all_inst_pred = list(all_inst_pred)
  
        inst_acc = accuracy_score(all_inst_label, all_inst_pred)
        print(classification_report(all_inst_label, all_inst_pred,zero_division=1))
        
        inst_p_macro = precision_score(all_inst_label, all_inst_pred, average='macro',zero_division=1)
        inst_r_macro = recall_score(all_inst_label, all_inst_pred, average='macro',zero_division=1)
        inst_f1_macro = f1_score(all_inst_label, all_inst_pred, average='macro',zero_division=1)

    # get inst_auc
    inst_aucs = []
    if not inst_refinement:
        for class_idx in range(n_classes):
            inst_score_sub = inst_probs[class_idx]
            if len(inst_score_sub)==0:
                continue
            fpr, tpr, _ = roc_curve(inst_binary_labels[class_idx], inst_score_sub)
            inst_aucs.append(calc_auc(fpr, tpr))
        fpr,tpr,_ = roc_curve(inst_binary_labels[n_classes], all_normal_score)
        inst_aucs.append(calc_auc(fpr, tpr))
    else:
        binary_inst_label = label_binarize(all_inst_label, classes=[i for i in range(n_classes+1)])
        for class_idx in range(n_classes+1):
            if class_idx in all_inst_label:
                inst_score_sub = all_inst_score[:,class_idx]
                if len(inst_score_sub)==0 or np.isnan(inst_score_sub).any():
                    continue
                fpr, tpr, _ = roc_curve(binary_inst_label[:, class_idx], inst_score_sub)
                inst_aucs.append(calc_auc(fpr, tpr))
            else:
                inst_aucs.append(float('nan'))
    inst_auc = np.nanmean(np.array(inst_aucs))
    
    aucs = []
    binary_labels = labels # wsis x cls
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))
    auc = np.nanmean(np.array(aucs))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        for i in range(n_classes):
            writer.add_scalar('val/auc_c{}'.format(str(i)), aucs[i], epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_auc', inst_auc, epoch)
        writer.add_scalar('val/inst_acc', inst_acc, epoch)
        writer.add_scalar('val/inst_p_macro', inst_p_macro, epoch)
        writer.add_scalar('val/inst_r_macro', inst_r_macro, epoch)
        writer.add_scalar('val/inst_f1_macro', inst_f1_macro, epoch)
        

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}'.
          format(val_loss, val_error, auc, inst_auc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count)) 
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
    
    # LR adjust
    scheduler.step(val_loss)
    
    torch.save(model.state_dict(), os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint_best.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary(model, loader, args):
    
    
    n_classes = args.n_classes
    model_type = args.model_type
    multi_label = args.multi_label
    inst_refinement = args.inst_refinement
    
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_pos = []
    all_inst_score_neg = []
    
    inst_probs = [[] for _ in range(n_classes+1)]
    inst_preds = [[] for _ in range(n_classes+1)]
    inst_binary_labels = [[] for _ in range(n_classes+1)]
    
    pos_accs_each_wsi = [[] for _ in range(n_classes)]
    neg_accs_each_wsi = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros((len(loader), n_classes))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
        
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        if multi_label:
            label = label.to(device)
            index_label = torch.nonzero(label.squeeze()).to(device)
        else:
            index_label = label.to(device)
            label = torch.zeros(n_classes)
            label[index_label.long()] = 1
            label = label.to(device)
        
        if inst_label!=[]:
            all_inst_label += inst_label
        
        slide_id = slide_ids.iloc[batch_idx]
            
        data = data.to(device)
        with torch.no_grad():
            mask = cors[1]
            sp = cors[2]
            adj = cors[3]
            score, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp, adj, index_label, 
                                                                  instance_eval=inst_refinement)

#         Y_prob=torch.mean(torch.stack(Y_prob),dim=0)
        Y_prob = Y_prob[0]
                
        if inst_label!=[]:
            label_hot = label_binarize(inst_label, classes=[i for i in range(n_classes+1)])
            inst_probs_tmp = [] 
            inst_preds_tmp = []
            if not inst_refinement:
                inst_score = score.detach().cpu().numpy()

                # max-min standard all_inst_score and all_inst_score_pos
                for class_idx in range(n_classes):
                    if class_idx not in index_label:
                        inst_score[:,class_idx] = [0]*len(inst_label)
                        continue

                    inst_score_one_class = inst_score[:,class_idx]
#                     if len(set(inst_score_one_class))>1:
                    inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                                                max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))
                    inst_score_one_class = list(inst_score_one_class)
                    inst_score[:,class_idx] = inst_score_one_class

                    inst_probs[class_idx]+=inst_score_one_class
                    inst_probs_tmp.append(inst_score_one_class)                  

                    inst_preds[class_idx]+=[0 if i<0.5 else 1 for i in inst_score_one_class]
                    inst_preds_tmp.append([0 if i<0.5 else 1 for i in inst_score_one_class])

                    inst_binary_labels[class_idx]+=list(label_hot[:,class_idx])

                if inst_preds_tmp:
                    inst_preds_tmp = np.mean(np.stack(inst_preds_tmp), axis=0)
                else:
                    inst_preds_tmp = [0]*len(inst_label)
                inst_preds_tmp = [1 if i==0 else 0 for i in inst_preds_tmp]
                inst_preds[n_classes] += inst_preds_tmp
                inst_binary_labels[n_classes]+=list(label_hot[:,n_classes])

                if inst_probs_tmp:
                    neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #三类平均，越低越是neg
#                     if len(set(neg_score))>1:
                    neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
#                     neg_score = list(neg_score)
                else:
                    neg_score = [0]*len(inst_label)



            else:
                inst_score = ref_score.detach().cpu().numpy()
                pos_score = score.detach().cpu().numpy() #final_score_sp

                # max-min standard all_inst_score and all_inst_score_pos
                for class_idx in range(n_classes):
#                     if class_idx not in Y_hat:
                    if class_idx not in index_label:
                        continue
                    inst_score_one_class = pos_score[:,class_idx]
#                     if len(set(inst_score_one_class))>1:
                    inst_score_one_class = list((inst_score_one_class-inst_score_one_class.min())/
                                                max(inst_score_one_class.max()-inst_score_one_class.min(),1e-10))

                    inst_probs_tmp.append(inst_score_one_class)


                if inst_probs_tmp:
                    neg_score = np.mean(np.stack(inst_probs_tmp), axis=0) #三类平均，越低越是neg
#                     if len(set(neg_score))>1:
                    neg_score = list((neg_score-neg_score.min())/max(neg_score.max()-neg_score.min(),1e-10))
#                     neg_score = list(neg_score)
                else:
                    neg_score = [0]*len(inst_label)
#                 neg_score = list((neg_score-neg_score.min())/(neg_score.max()-neg_score.min()))


            all_inst_score.append(inst_score)
            all_inst_score_neg += neg_score

            
        acc_logger.log(Y_hat, index_label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.squeeze().detach().cpu().numpy()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 
                                           'label': torch.nonzero(label.squeeze()).squeeze().cpu().numpy()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    
    # calculate inst_auc and inst_acc   
    
    all_inst_score = np.concatenate(all_inst_score,axis=0)
    all_normal_score = [1-x for x in all_inst_score_neg] #转化为是normal类的概率 越高越好
    
    # get inst_pred inst_acc
    if not inst_refinement:
        inst_accs = []
        for i in range(n_classes+1):
            if len(inst_binary_labels[i])==0:
                continue
            inst_accs.append(accuracy_score(inst_binary_labels[i], inst_preds[i]))
            inst_acc = np.mean(inst_accs)
            print('class {}'.format(str(i)))
            print(classification_report(inst_binary_labels[i], inst_preds[i],zero_division=1))
    else:
        all_inst_pred = np.argmax(all_inst_score,axis=1)
        all_inst_pred = list(all_inst_pred)
  
        inst_acc = accuracy_score(all_inst_label, all_inst_pred)
        print(classification_report(all_inst_label, all_inst_pred,zero_division=1))
    
    # get inst_auc
    inst_aucs = []
    if not inst_refinement:
        for class_idx in range(n_classes):
            inst_score_sub = inst_probs[class_idx]
            if len(inst_score_sub)==0:
                continue
            fpr, tpr, _ = roc_curve(inst_binary_labels[class_idx], inst_score_sub)
            inst_aucs.append(calc_auc(fpr, tpr))
        fpr,tpr,_ = roc_curve(inst_binary_labels[n_classes], all_normal_score)
        inst_aucs.append(calc_auc(fpr, tpr))
    else:
        binary_inst_label = label_binarize(all_inst_label, classes=[i for i in range(n_classes+1)])
        for class_idx in range(n_classes+1):
            if class_idx in all_inst_label:
                inst_score_sub = all_inst_score[:,class_idx]
                fpr, tpr, _ = roc_curve(binary_inst_label[:, class_idx], inst_score_sub)
                inst_aucs.append(calc_auc(fpr, tpr))
            else:
                inst_aucs.append(float('nan'))
    inst_auc = np.nanmean(np.array(inst_aucs))


    aucs = []
    binary_labels = all_labels
    for class_idx in range(n_classes):
        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
        aucs.append(calc_auc(fpr, tpr))

    auc = np.nanmean(np.array(aucs))


    return patient_results, test_error, auc, inst_auc, acc_logger
