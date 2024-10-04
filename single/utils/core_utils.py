import numpy as np
import torch
from utils.utils import *
import os
import pandas as pd
from torch.optim import lr_scheduler
from datasets.dataset_nic import save_splits
from models.model_smmile import RAMIL, SMMILe, SMMILe_SINGLE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.bi_tempered_loss_pytorch import bi_tempered_binary_logistic_loss

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
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
    def __init__(self, patience=20, stop_epoch=80, verbose=False):
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

    if args.bag_loss == 'bibce':
        args.bi_loss = True
        
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {'dropout': args.drop_out, 'drop_rate': args.drop_rate, 'n_classes': args.n_classes, 
                  'fea_dim': args.fea_dim, "size_arg": args.model_size, 'n_refs': args.n_refs}

    if args.model_type == 'ramil':
        model = RAMIL(**model_dict)
    elif args.model_type == 'smmile':
        model = SMMILe(**model_dict)
    elif args.model_type == 'smmile_single':
        model = SMMILe_SINGLE(**model_dict)
    else:
        raise NotImplementedError

    if args.models_dir is not None:
        ckpt_path = os.path.join(args.models_dir, 's_{}_checkpoint_best.pt'.format(cur))
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt, strict=False)
            print('\nThe model has been loaded from %s' % ckpt_path)
        else:
            print('\nThe model will train from scrash')
    
    model.relocate()
    print('Done!')
    print_network(model)

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
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)
    else:
        early_stopping = None
    print('Done!')
    
    _, val_error, val_auc, val_iauc, _ = summary(model, val_loader, args)

    if args.ref_start_epoch == 0 and args.inst_refinement: # continue training the model from ckpt 
        ref_start = True
    else:
        ref_start = False
    
    for epoch in range(args.max_epochs):
        if args.model_type in ['ramil','smmile']:
            train_loop_smmile(epoch, model, train_loader, optimizer, writer, loss_fn, ref_start, args)
            stop = validate_smmile(cur, epoch, model, val_loader, early_stopping, writer, loss_fn, ref_start, args, scheduler, mode='val')
            # _ = validate_smmile(cur, epoch, model, test_loader, early_stopping, writer, loss_fn, ref_start, args, mode='test')
        elif args.model_type in ['smmile_single']:
            train_loop_smmile_single(epoch, model, train_loader, optimizer, writer, loss_fn, ref_start, args)
            stop = validate_smmile_single(cur, epoch, model, val_loader, early_stopping, writer, loss_fn, ref_start, args, scheduler, mode='val')
            # _ = validate_smmile(cur, epoch, model, test_loader, early_stopping, writer, loss_fn, ref_start, args, mode='test')
        else:
            raise NotImplementedError
        
        if (stop and not ref_start and args.inst_refinement) or (epoch == args.ref_start_epoch and args.inst_refinement):
            ref_start = True
            early_stopping = EarlyStopping(patience = 50, stop_epoch=100, verbose = True)
        elif stop:
            break

    if args.early_stopping: # load the best model to evaluate
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint_best.pt".format(cur))))
    else: # save the current model
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
        
def train_loop_smmile(epoch, model, loader, optimizer, writer = None, loss_fn = None, ref_start = False, args = None): 
    
    n_classes = args.n_classes
    bi_loss = args.bi_loss
    drop_with_score = args.drop_with_score
    D = args.D
    superpixel = args.superpixel
    sp_smooth = args.sp_smooth
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
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    all_inst_score_pos = []
    all_inst_score_neg = []
    
    if not ref_start:
        inst_refinement = False

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
        
        label = label.to(device)
        wsi_label = torch.zeros(n_classes)
        wsi_label[label.long()] = 1
        wsi_label = wsi_label.to(device)
        
        total_loss = 0
        total_loss_value = 0
        inst_loss_value = 0
        mrf_loss_value = 0
        
        # for data_index in range(len(data)):
        data = data.to(device)
        mask = cors[1]
        sp = cors[2]
        adj = cors[3]

        score, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp, adj, label, 
                                                              group_numbers = G, 
                                                              superpixels = superpixel,
                                                              sp_smooth = sp_smooth,
                                                              drop_with_score=drop_with_score,
                                                              drop_times = D,
                                                              instance_eval=inst_refinement, 
                                                              inst_rate=inst_rate,
                                                              mrf=mrf, 
                                                              tau=tau)

        # statistics for instance
        if inst_label!=[] and sum(inst_label)!=0:

            inst_label = [1 if patch_label>0 else patch_label for patch_label in inst_label] # normal vs cancer, keep -1
            all_inst_label += inst_label

            inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
            inst_score = list((inst_score-inst_score.min())/max((inst_score.max()-inst_score.min()),1e-10))
            inst_pred = [1 if i>0.5 else 0 for i in inst_score]
            
            pos_score = score[:,label].detach().cpu().numpy()[:,0]
            pos_score = list((pos_score-pos_score.min())/max((pos_score.max()-pos_score.min()),1e-10))

            neg_score = torch.mean(score, dim=-1).detach().cpu().numpy()
            neg_score = list((neg_score-neg_score.min())/max((neg_score.max()-neg_score.min()),1e-10))
                
            if inst_refinement:
                inst_score = list(1 - ref_score[:,-1].detach().cpu().numpy())
                inst_pred = torch.argmax(ref_score, dim=1).detach().cpu().numpy()
                inst_pred = [0 if i==n_classes else 1 for i in inst_pred] # for one-class cancer

                
            all_inst_score += inst_score
            all_inst_pred += inst_pred

            # record pos & neg acc 
            df_score = pd.DataFrame([inst_label, pos_score, neg_score]).T
    
            df_score = df_score.sort_values(by=1)
            df_score_top = df_score.iloc[-int(df_score.shape[0]*inst_rate):,:]
            df_score_top = df_score_top[df_score_top[0]!=-1]
            
            df_score = df_score.sort_values(by=2)
            df_score_down = df_score.iloc[:int(df_score.shape[0]*inst_rate),:]
            df_score_down = df_score_down[df_score_down[0]!=-1]
            
            if not df_score_top.empty:
                all_inst_score_pos += df_score_top[0].tolist()
            if not df_score_down.empty:
                all_inst_score_neg += df_score_down[0].tolist()
        
        acc_logger.log(Y_hat, label)

        loss = loss_fn(Y_prob[0], wsi_label)/len(Y_prob)

        for one_prob in Y_prob[1:]:
            if bi_loss:
                loss += bi_tempered_binary_logistic_loss(one_prob, wsi_label, 0.2, 1., reduction='mean')/len(Y_prob)
            else:
                loss += loss_fn(one_prob, wsi_label)/len(Y_prob)
        
        loss_value = loss.item()
    
        total_loss += loss
        total_loss_value += loss_value

        if inst_refinement:
            instance_loss = results_dict['instance_loss']
            if instance_loss>0:
                total_loss += instance_loss 
                inst_loss_value += instance_loss.item()
        
        if mrf:
            mrf_loss = results_dict['mrf_loss']
            if mrf_loss>0:
                total_loss += mrf_loss 
                mrf_loss_value += mrf_loss.item()
            

        error = calculate_error(Y_hat, label)
        train_error += error
        
        train_loss += total_loss_value
        inst_loss += inst_loss_value
        m_loss += mrf_loss_value

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
    
    # excluding -1 
    all_inst_label = np.array(all_inst_label)
    all_inst_score = np.array(all_inst_score)
    # all_inst_score_pos = np.array(all_inst_score_pos)
    # all_inst_score_neg = np.array(all_inst_score_pos)
    all_inst_pred = np.array(all_inst_pred)
    
    all_inst_score = all_inst_score[all_inst_label!=-1]
    # all_inst_score_pos = all_inst_score_pos[all_inst_label!=-1]
    # all_inst_score_neg = all_inst_score_neg[all_inst_label!=-1]
    all_inst_pred = all_inst_pred[all_inst_label!=-1]
    
    all_inst_label = all_inst_label[all_inst_label!=-1]
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    # df_score = pd.DataFrame([all_inst_label, all_inst_score, all_inst_score_pos, all_inst_score_neg]).T
    
    # df_score = df_score.sort_values(by=2)
    # df_score[2] = df_score[2].apply(lambda x: 1 if x>0.5 else 0)
    # df_score_top = df_score.iloc[-int(df_score.shape[0]*inst_rate):,:]
    
    # df_score = df_score.sort_values(by=3)
    # df_score[3] = df_score[3].apply(lambda x: 1 if x>0.5 else 0)
    # df_score_down = df_score.iloc[:int(df_score.shape[0]*inst_rate),:]
    
    pos_acc = (sum(all_inst_score_pos)/len(all_inst_score_pos))
    neg_acc = (1-sum(all_inst_score_neg)/len(all_inst_score_neg))
    print("seleted pos %f acc: %f" % (inst_rate, pos_acc))
    print("seleted neg %f acc: %f" % (inst_rate, neg_acc))
    
    # all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_pred)
    print(classification_report(all_inst_label, all_inst_pred, zero_division=1))

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, mrf_loss: {:.4f}, inst_loss: {:.4f}, inst_acc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc, m_loss, inst_loss, inst_acc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/pos_acc', pos_acc, epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)

def train_loop_smmile_single(epoch, model, loader, optimizer, writer = None, loss_fn = None, ref_start = False, args = None):

    n_classes = args.n_classes
    bi_loss = args.bi_loss
    consistency = args.consistency
    drop_with_score = args.drop_with_score
    D = args.D
    superpixel = args.superpixel
    sp_smooth = args.sp_smooth
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
    all_inst_score_pos = []
    
    if not ref_start:
        inst_refinement = False

    print('\n')
    for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            
        total_loss = 0
        total_loss_value = 0
        inst_loss_value = 0
        mrf_loss_value = 0
        consist_loss_value = 0
        
        label = label.to(device, non_blocking=True).to(torch.float32)

        data = data.to(device)
        mask = cors[1]
        sp = cors[2]
        adj = cors[3]
        
        _, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp, adj, label=label, 
                                                          group_numbers = G,
                                                          superpixels = superpixel, 
                                                          sp_smooth = sp_smooth,
                                                          drop_with_score=drop_with_score, 
                                                          drop_times = D,
                                                          instance_eval=inst_refinement, 
                                                          inst_rate=inst_rate,
                                                          mrf=mrf, 
                                                          tau=tau,
                                                          consistency = consistency)
        
        acc_logger.log(Y_hat, label)

        loss = loss_fn(Y_prob[0], label.float())
        
        for one_prob in Y_prob[1:]:
            if bi_loss:
                loss += bi_tempered_binary_logistic_loss(one_prob, label, 0.2, 1., reduction='mean')/len(Y_prob[1:])
            else:
                loss += loss_fn(one_prob, label.float())/len(Y_prob[1:])

        loss_value = loss.item()
    
        total_loss += loss
        total_loss_value += loss_value

        
        instance_loss = results_dict['instance_loss']
        if instance_loss>0:
            total_loss += instance_loss 
            inst_loss_value += instance_loss.item()
        
        
        mrf_loss = results_dict['mrf_loss'] 
        if mrf_loss>0:
            total_loss += mrf_loss 
            mrf_loss_value += mrf_loss.item()
                
        consist_loss = results_dict['consist_loss']
        if consist_loss>0:
            total_loss += consist_loss
            consist_loss_value += consist_loss.item()

        error = calculate_error(Y_hat, label)
        train_error += error
        
        train_loss += total_loss_value
        inst_loss += inst_loss_value
        mrf_loss += mrf_loss_value
        cons_loss += consist_loss_value

        # backward pass
        total_loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()
        
        if inst_label!=[] and sum(inst_label)!=0:

            all_inst_label += inst_label

            if not inst_refinement:
                inst_score = ref_score[:, 0].detach().cpu().numpy()
                inst_score = list((inst_score-inst_score.min())/(inst_score.max()-inst_score.min()))
                inst_pred = [1 if i>0.5 else 0 for i in inst_score]
                
                pos_score = ref_score[:, 0].detach().cpu().numpy()
                pos_score = list((pos_score-pos_score.min())/(pos_score.max()-pos_score.min()))
                
            else:
                inst_score = list(ref_score[:, 1].detach().cpu().numpy())
                inst_pred = list(torch.argmax(ref_score, dim=1).detach().cpu().numpy())
                
                pos_score = list(ref_score[:, 1].detach().cpu().numpy())

                
            all_inst_score += inst_score
            all_inst_pred += inst_pred
            
            all_inst_score_pos += pos_score

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)
    inst_loss /= len(loader)
    m_loss /= len(loader)
    cons_loss /= len(loader)
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    df_score = pd.DataFrame([all_inst_label, all_inst_score, all_inst_score_pos]).T
    
    df_score = df_score.sort_values(by=2)
    df_score[2] = df_score[2].apply(lambda x: 1 if x>0.5 else 0)
    df_score_top = df_score[df_score[2] == 1]
    # df_score_top = df_score.iloc[-int(df_score.shape[0]*inst_rate):,:]
    
    df_score = df_score.sort_values(by=1)
    df_score[1] = df_score[1].apply(lambda x: 1 if x>0.5 else 0)
    df_score_down = df_score.iloc[:int(df_score.shape[0]*inst_rate),:]
    
    # pos_acc_all = (df_score_top_all[0].sum()/df_score_top_all.shape[0])
    pos_acc = (df_score_top[0].sum()/df_score_top.shape[0])
    neg_acc = (1-df_score_down[0].sum()/df_score_down.shape[0])
    # print("seleted pos all acc: %f" % pos_acc_all)
    print("seleted pos %f acc: %f" % (inst_rate, pos_acc))
    print("seleted neg %f acc: %f" % (inst_rate, neg_acc))
    
    # all_inst_score = [1 if i>0.5 else 0 for i in all_inst_score]
    inst_acc = accuracy_score(all_inst_label, all_inst_pred)
    print(classification_report(all_inst_label, all_inst_pred, zero_division=1))

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}, inst_auc: {:.4f}, m_loss: {:.4f}, inst_loss: {:.4f}, cons_loss: {:.4f}, inst_acc: {:.4f}'.format(epoch, train_loss, train_error, inst_auc, m_loss, inst_loss, cons_loss, inst_acc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/pos_acc', pos_acc, epoch)
        writer.add_scalar('train/neg_acc', neg_acc, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/inst_auc', inst_auc, epoch)

def validate_smmile(cur, epoch, model, loader, early_stopping = None, writer = None, loss_fn = None, ref_start=False, args=None, 
                    scheduler=None, mode='val'):
    
    n_classes = args.n_classes
    bi_loss = args.bi_loss
    superpixel = args.superpixel
    sp_smooth = args.sp_smooth
    G = args.G
    inst_refinement = args.inst_refinement
    results_dir = args.results_dir
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    inst_loss = 0.
    val_error = 0.
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    
    if not ref_start:
        inst_refinement = False
    
    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            label = label.to(device, non_blocking=True)
            wsi_label = torch.zeros(n_classes)
            wsi_label[label.long()] = 1
            wsi_label = wsi_label.to(device)
            
            if inst_label!=[] and sum(inst_label)!=0:
                inst_label = [1 if patch_label>0 else patch_label for patch_label in inst_label] # normal vs cancer, keep -1
                all_inst_label += inst_label
            
            data = data.to(device)
            mask = cors[1]
            sp = cors[2]
            adj = cors[3]

            score, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp, adj, label, 
                                                                  superpixels = superpixel,
                                                                  sp_smooth = sp_smooth,
                                                                  group_numbers = G,
                                                                  instance_eval=inst_refinement)
            
            if inst_label!=[] and sum(inst_label)!=0:
                if not inst_refinement:
                    inst_score = score[:,Y_hat].detach().cpu().numpy()[:,0]
                    inst_score = list((inst_score-inst_score.min())/max((inst_score.max()-inst_score.min()), 1e-10))
                    inst_pred = [1 if i>0.5 else 0 for i in inst_score]
                else:
                    inst_score = list(1 - ref_score[:,-1].detach().cpu().numpy())
                    inst_pred = torch.argmax(ref_score, dim=1).detach().cpu().numpy()
                    inst_pred = [0 if i==n_classes else 1 for i in inst_pred] # for one-class cancer
                all_inst_score += inst_score
                all_inst_pred += inst_pred
            
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(Y_prob[0], wsi_label)/len(Y_prob)

            for one_prob in Y_prob[1:]:
                if bi_loss:
                    loss += bi_tempered_binary_logistic_loss(one_prob, wsi_label, 0.2, 1., reduction='mean')/len(Y_prob)
                else:
                    loss += loss_fn(one_prob, wsi_label)/len(Y_prob)

            Y_prob = Y_prob[0]
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            
            if inst_refinement:
                instance_loss = results_dict['instance_loss']
                if instance_loss>0:
                    val_loss += instance_loss.item()
                    inst_loss += instance_loss.item()

            

    val_error /= len(loader)
    val_loss /= len(loader)
    inst_loss /= len(loader)
    
    # excluding -1 
    all_inst_label = np.array(all_inst_label)
    all_inst_score = np.array(all_inst_score)
    all_inst_pred = np.array(all_inst_pred)
    
    all_inst_score = all_inst_score[all_inst_label!=-1]
    all_inst_pred = all_inst_pred[all_inst_label!=-1]
    all_inst_label = all_inst_label[all_inst_label!=-1]
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)

    inst_acc = accuracy_score(all_inst_label, all_inst_pred)
    
    inst_p_macro = precision_score(all_inst_label, all_inst_pred, average='macro')
#     inst_p_micro = precision_score(all_inst_label, all_inst_pred, average='micro')
#     inst_p_weighted = precision_score(all_inst_label, all_inst_pred, average='weighted')

    inst_r_macro = recall_score(all_inst_label, all_inst_pred, average='macro')
#     inst_r_micro = recall_score(all_inst_label, all_inst_pred, average='micro')
#     inst_r_weighted = recall_score(all_inst_label, all_inst_pred, average='weighted')

    inst_f1_macro = f1_score(all_inst_label, all_inst_pred, average='macro')
#     inst_f1_micro = f1_score(all_inst_label, all_inst_pred, average='micro')
#     inst_f1_weighted = f1_score(all_inst_label, all_inst_pred, average='weighted')
    
    print(classification_report(all_inst_label, all_inst_pred, zero_division=1))

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    if writer:
        writer.add_scalar('{}/loss'.format(mode), val_loss, epoch)
        writer.add_scalar('{}/inst_loss'.format(mode), inst_loss, epoch)
        writer.add_scalar('{}/auc'.format(mode), auc, epoch)
        writer.add_scalar('{}/error'.format(mode), val_error, epoch)
        writer.add_scalar('{}/inst_acc'.format(mode), inst_acc, epoch)
        writer.add_scalar('{}/inst_auc'.format(mode), inst_auc, epoch)
        writer.add_scalar('{}/inst_p_macro'.format(mode), inst_p_macro, epoch)
        writer.add_scalar('{}/inst_r_macro'.format(mode), inst_r_macro, epoch)
        writer.add_scalar('{}/inst_f1_macro'.format(mode), inst_f1_macro, epoch)

    print('\n {} Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}'.format(mode, val_loss, val_error, auc, inst_auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     
    
    if mode == 'val':
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

def validate_smmile_single(cur, epoch, model, loader, early_stopping = None, writer = None, loss_fn = None, 
                          ref_start=False, args=None, scheduler=None, mode='val'):
    
    n_classes = args.n_classes
    bi_loss = args.bi_loss
    superpixel = args.superpixel
    sp_smooth = args.sp_smooth
    G = args.G
    inst_refinement = args.inst_refinement
    results_dir = args.results_dir

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    inst_loss = 0.
    val_error = 0.
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []
    
    prob = np.zeros((len(loader), 1))
    labels = np.zeros(len(loader))
    
    if not ref_start:
        inst_refinement = False

    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):

            label = label.to(device, non_blocking=True).to(torch.float32)

            data = data.to(device)
            mask = cors[1]
            sp = cors[2]
            adj = cors[3]

            score, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp, adj, label=label, 
                                                                  superpixels = superpixel,
                                                                  sp_smooth = sp_smooth,
                                                                  group_numbers = G, 
                                                                  instance_eval=inst_refinement)

            if inst_label!=[] and sum(inst_label)!=0:

                all_inst_label += inst_label

                if not inst_refinement:
                    inst_score = score[:, 0].detach().cpu().numpy()
                    inst_score = list((inst_score-inst_score.min())/(inst_score.max()-inst_score.min()))
                    inst_pred = [1 if i>0.5 else 0 for i in inst_score]

                else:
                    inst_score = list(ref_score[:, 1].detach().cpu().numpy())
                    inst_pred = list(torch.argmax(ref_score, dim=1).detach().cpu().numpy())

                all_inst_score += inst_score
                all_inst_pred += inst_pred
        
            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(Y_prob[0], label.float())
            
            for one_prob in Y_prob[1:]:
                if bi_loss:
                    loss += bi_tempered_binary_logistic_loss(one_prob, label, 0.2, 1., reduction='mean')/len(Y_prob[1:])
                else:
                    loss += loss_fn(one_prob, label.float())/len(Y_prob[1:])

            Y_prob = Y_prob[0]
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            
            if inst_refinement:
                instance_loss = results_dict['instance_loss']
                if instance_loss!=0:
                    val_loss += instance_loss.item()
                    inst_loss += instance_loss.item()
            

    val_error /= len(loader)
    val_loss /= len(loader)
    inst_loss /= len(loader)
    
    inst_auc = roc_auc_score(all_inst_label, all_inst_score)

    inst_acc = accuracy_score(all_inst_label, all_inst_pred)
    
    inst_p_macro = precision_score(all_inst_label, all_inst_pred, average='macro')
    inst_p_micro = precision_score(all_inst_label, all_inst_pred, average='micro')
    inst_p_weighted = precision_score(all_inst_label, all_inst_pred, average='weighted')

    inst_r_macro = recall_score(all_inst_label, all_inst_pred, average='macro')
    inst_r_micro = recall_score(all_inst_label, all_inst_pred, average='micro')
    inst_r_weighted = recall_score(all_inst_label, all_inst_pred, average='weighted')

    inst_f1_macro = f1_score(all_inst_label, all_inst_pred, average='macro')
    inst_f1_micro = f1_score(all_inst_label, all_inst_pred, average='micro')
    inst_f1_weighted = f1_score(all_inst_label, all_inst_pred, average='weighted')

    print(classification_report(all_inst_label, all_inst_pred, zero_division=1))

    auc = roc_auc_score(labels, prob[:,0])

    if writer:
        writer.add_scalar('{}/loss'.format(mode), val_loss, epoch)
        writer.add_scalar('{}/inst_loss'.format(mode), inst_loss, epoch)
        writer.add_scalar('{}/auc'.format(mode), auc, epoch)
        writer.add_scalar('{}/error'.format(mode), val_error, epoch)
        writer.add_scalar('{}/inst_acc'.format(mode), inst_acc, epoch)
        writer.add_scalar('{}/inst_auc'.format(mode), inst_auc, epoch)
        writer.add_scalar('{}/inst_p_macro'.format(mode), inst_p_macro, epoch)
        writer.add_scalar('{}/inst_r_macro'.format(mode), inst_r_macro, epoch)
        writer.add_scalar('{}/inst_f1_macro'.format(mode), inst_f1_macro, epoch)
        

    print('\n {} Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, inst_auc: {:.4f}'.format(mode, val_loss, val_error, auc, inst_auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     
    
    
    if mode == 'val':
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
    inst_refinement = args.inst_refinement
    superpixel = args.superpixel
    sp_smooth = args.sp_smooth
    G = args.G
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    all_inst_label = []
    all_inst_score = []
    all_inst_pred = []

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    
    with torch.no_grad():
        for batch_idx, (data, label, cors, inst_label) in enumerate(loader):
            data = data.to(device)
            label = label.to(device, non_blocking=True)

            slide_id = slide_ids.iloc[batch_idx]

            mask = cors[1]
            sp = cors[2]
            adj = cors[3]

            score, Y_prob, Y_hat, ref_score, results_dict = model(data, mask, sp, adj, label, 
                                                                  superpixels = superpixel,
                                                                  sp_smooth = sp_smooth,
                                                                  group_numbers = G,
                                                                  instance_eval=inst_refinement)
            
            Y_prob = Y_prob[0]

            if inst_label!=[] and sum(inst_label)!=0:
                
                inst_label = [1 if patch_label>0 else patch_label for patch_label in inst_label] # normal vs cancer, keep -1
                all_inst_label += inst_label

                if not inst_refinement:
                    if model_type == 'smmile':
                        inst_score = score[:,Y_hat].detach().cpu().numpy()[:]
                    elif model_type == 'smmile_single':
                        inst_score = score[:, 0].detach().cpu().numpy()[:]
                    else:
                        inst_score = score[:,Y_hat].detach().cpu().numpy()[:]
                    inst_score = list((inst_score-inst_score.min())/max((inst_score.max()-inst_score.min()), 1e-10))
                    inst_pred = [1 if i>0.5 else 0 for i in inst_score]
                else:
                    if model_type == 'smmile':
                        inst_score = list(1 - ref_score[:,-1].detach().cpu().numpy())
                        inst_pred = torch.argmax(ref_score, dim=1).detach().cpu().numpy()
                        inst_pred = [0 if i==n_classes else 1 for i in inst_pred] # for one-class cancer
                    elif model_type == 'smmile_single':
                        inst_score = list(ref_score[:,1].detach().cpu().numpy())
                        inst_pred = list(torch.argmax(ref_score, dim=1).detach().cpu().numpy())
                    else:
                        inst_score = list(1 - ref_score[:,-1].detach().cpu().numpy())
                        inst_pred = torch.argmax(ref_score, dim=1).detach().cpu().numpy()
                        inst_pred = [0 if i==n_classes else 1 for i in inst_pred] # for one-class cancer

                all_inst_score += inst_score
                all_inst_pred += inst_pred
            
            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()

            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error

    test_error /= len(loader)
    
    # excluding -1 
    all_inst_label = np.array(all_inst_label)
    all_inst_score = np.array(all_inst_score)
    all_inst_pred = np.array(all_inst_pred)
    
    all_inst_score = all_inst_score[all_inst_label!=-1]
    all_inst_pred = all_inst_pred[all_inst_label!=-1]
    all_inst_label = all_inst_label[all_inst_label!=-1]

    inst_auc = roc_auc_score(all_inst_label, all_inst_score)
    print("inst level aucroc: %f" % inst_auc)
    inst_acc = accuracy_score(all_inst_label, all_inst_pred)
    
    print(classification_report(all_inst_label, all_inst_pred, zero_division=1))

    if model_type == 'smmile_single':
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        aucs = []
        
        if n_classes == 2:
            binary_labels = label_binarize(all_labels, classes=[0,1,2])[:,:n_classes]
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, inst_auc, acc_logger