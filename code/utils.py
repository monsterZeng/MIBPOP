import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, accuracy_score

def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    if cuda:
        torch.cuda.manual_seed_all(seed)


def compute_accuracy(y_pred, y_target):
    """
        TN: the number of true negatives
        TP: the number of true positives
        FN: the number of false negatives 
        FP: the number of false positives
    """
    
    _, y_pred_indices = torch.max(y_pred, dim=1)
    y_pred_indices = y_pred_indices.detach().cpu().numpy()
    
    y_target = y_target.squeeze().detach().cpu().numpy()
    accuracy = accuracy_score(y_target, y_pred_indices)
    recall = recall_score(y_target, y_pred_indices)
    precision = precision_score(y_target, y_pred_indices)
    #return accuracy, recall, precision
    # n_correct = (y_pred_indices & y_target).sum().item()
    # acc = n_correct / len(y_pred_indices) * 100
    # TP = (y_pred_indices == 0) & (y_target == 0).sum().item()
    # FP = ((y_pred_indices == 0) & (y_target == 1)).sum().item() 
    
    TN = ((y_pred_indices == 0) & (y_target == 0)).sum().item()
    FN = ((y_pred_indices == 0) & (y_target == 1)).sum().item() 
    

    # Sp, Sn = TN / (TN + FP), TP / (TP + FN)
    
    return accuracy, TN / (TN + FN), recall

def fit(args, model, graphs, smiles):
    graphs = graphs.to(args.device)
    node_feats = graphs.ndata.pop('h').to(args.device)
    #print("node_feats.shape = ",node_feats.shape)
    return model(graphs, node_feats, smiles)

def train_func(args, th_epoch, data_loader, model, loss_func, optimizer):
    running_loss    = 0.0
    running_acc     = 0.0
    running_sp_acc  = 0.0
    running_sn_acc  = 0.0
    for idx, batch_data in enumerate(data_loader):
        # step 1: 准备数据
        smiles, graphs, labels, masks = batch_data
        graphs, labels, masks = graphs.to(args.device), labels.long().to(args.device), masks.to(args.device)
        # step 2: 拟合数据
        #logits, readout, att = fit(args, model, graphs)
        logits1, logits2, logits3 = fit(args, model, graphs, smiles)
        # step 3: 求损失
        loss1 = loss_func(logits1, labels.squeeze())
        loss2 = loss_func(logits2, labels.squeeze())
        loss3 = loss_func(logits3, labels.squeeze())
        
        loss = (loss3 + loss2 + loss1)/3
        optimizer.zero_grad()
        # step 4: 求梯度
        loss.backward()
        # step 5: 进行优化
        optimizer.step()
        #train_meter.update(logits, labels, masks)
        # step 6: compute running loss & running loss
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (idx + 1)
        
        # if th_epoch > 5:
        #     acc_t, sp_acc_t, sn_acc_t = compute_accuracy(logits1, labels)
        #     running_acc += (acc_t - running_acc) / (idx + 1)
        #     running_sp_acc += (sp_acc_t - running_sp_acc) / (idx + 1)
        #     running_sn_acc += (sn_acc_t - running_sn_acc) / (idx + 1)

    return running_loss, running_acc, running_sp_acc, running_sn_acc

def evaluation(args, th_epoch, data_loader, model, loss_criterion):
    
    running_acc = 0.0
    running_loss = 0.0
    running_sp_acc = 0.0
    running_sn_acc = 0.0
    for idx, batch_data in enumerate(data_loader):
        # step 1: 准备数据
        smiles, graphs, labels, masks = batch_data
        graph, labels, masks = graphs.to(args.device), labels.long().to(args.device), masks.to(args.device)
        # step 2: 拟合数据
        #logits, readout, att = fit(args, model, graphs)
        logits = fit(args, model, graphs, smiles)
        # step 3: 求损失
        loss = loss_criterion(logits, labels.squeeze())
        #eval_meter.update(logits, labels, masks)
        
        # step 6: compute running loss & running loss
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (idx + 1)
        
        if th_epoch > 5:
            acc_t, sp_acc_t, sn_acc_t = compute_accuracy(logits, labels)
            running_acc += (acc_t - running_acc) / (idx + 1)
            running_sp_acc += (sp_acc_t - running_sp_acc) / (idx + 1)
            running_sn_acc += (sn_acc_t - running_sn_acc) / (idx + 1)

    return running_loss, running_acc, running_sp_acc, running_sn_acc

def vote(logits, y_target, device):
    """
    logits: list of logit
        logis.shape = [batch_size, 2]
    y_target: list of lable
        y_target.shape = [batch, 1]
    """
    y_target = y_target.squeeze()
    y_target = (y_target == 1)
    y_target = y_target.detach().cpu().numpy()
    preds = torch.LongTensor(logits[0].shape[0], len(logits)).to(device) 
    for i, logit in enumerate(logits):
        _, y_pred_indices = torch.max(logit, dim=1)
        preds[:, i] = y_pred_indices
    preds = torch.sum(preds, dim=1)

    preds = preds >= 3 # 大于三个投票为1，那么就是1，否则就是0
    preds = preds.detach().cpu().numpy()
    
    
    accuracy = accuracy_score(y_target, preds)
    recall = recall_score(y_target, preds)
    precision = precision_score(y_target, preds)
    TN = ((preds == 0) & (y_target == 0)).sum().item()
    FN = ((preds == 0) & (y_target == 1)).sum().item() 
    return accuracy, TN / (TN + FN), recall
    return accuracy, recall, precision  
    # correct_idx = torch.eq(preds, y_target)
    # n_correct = torch.eq(preds, y_target).sum().item()
    # TP = ((y_pred_indices == 0) & (y_target == 0)).sum().item()
    # FP =((y_pred_indices == 0) & (y_target == 1)).sum().item() 
    
    # TN = ((y_pred_indices == 1) & (y_target == 1)).sum().item()
    # FN =((y_pred_indices == 1) & (y_target == 0)).sum().item() 
    
    # Sp, Sn = TN / (TN + FP), TP / (TP + FN)
    
    #rb_acc = ((y_pred_indices == False) & (y_target == False)).sum().item() / torch.sum(y_pred_indices == False).item() * 100
    #nrb_acc = ((y_pred_indices == True) & (y_target == True)).sum().item() / torch.sum(y_pred_indices == True).item() * 100
    #return correct_idx, n_correct, Sp, Sn

