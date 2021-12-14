from dataset import get_dataloader
from utils import set_seed_everywhere,compute_accuracy, train_func, evaluation, fit, vote
from config import args
import torch
from collections import defaultdict, Counter
import pandas as pd
from sklearn.utils import shuffle
from model import MyGAT
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import math
from loss_func import FocalLoss
from GIN import MyGIN
# 设置随机种子
set_seed_everywhere(args.seed, torch.cuda.is_available())

# 获取全量数据集 Train+Test
total_data = defaultdict(list)
total_raw_data = pd.read_excel(args.raw_data_path, sheet_name=args.train_sheet_name)
total_data['Status'] = total_raw_data['Status']
total_data['Smiles'] = total_raw_data['Smiles']
total_data['class']  = total_raw_data['Class'] 
total_data = pd.DataFrame(total_data)
total_data = shuffle(total_data, random_state=args.seed)
total_data['class'] = total_data['class'].apply(lambda x: int(0) if x == 'RB' else int(1))

# 制作训练集
train_data = total_data[total_data['Status'] == "Train"]
train_data.drop("Status", axis = 1, inplace = True)

# 制作测试集
test_data = total_data[total_data['Status'] == "Test"]
test_data.drop("Status", axis = 1, inplace=True)

# 制作测试集
valid_data = defaultdict(list)
valid_raw_data = pd.read_excel(args.raw_data_path, sheet_name=args.valid_sheet_name)
valid_data['Smiles'] = valid_raw_data['Smiles']
valid_data['class']  = valid_raw_data['Class'] 
valid_data = pd.DataFrame(valid_data)
valid_data['class'] = valid_data['class'].apply(lambda x: int(0) if x == 'RB' else int(1))

kf = StratifiedKFold(n_splits = 5)
X_idx = np.array(list(range(len(train_data))))
Y_idx = np.array(train_data['class'])

i = 0
train_running_loss_s    = [[],[],[],[],[]]
train_running_acc_s     = [[],[],[],[],[]]
train_running_sp_acc_s  = [[],[],[],[],[]]
train_running_sn_acc_s  = [[],[],[],[],[]]

test_running_loss_s     = [[],[],[],[],[]]
test_running_acc_s      = [[],[],[],[],[]]
test_running_sp_acc_s   = [[],[],[],[],[]]
test_running_sn_acc_s   = [[],[],[],[],[]]

class_nums = Counter(train_data['class'])
weights = torch.FloatTensor([1 / v for k, v in class_nums.items()])
weights = weights.to(args.device)

for train_idx, test_idx in kf.split(X_idx, Y_idx):
    train = train_data.iloc[train_idx]
    test  = train_data.iloc[test_idx]
    train_dataloader = get_dataloader(train, args.train_batch_size)
    test_dataloader  = get_dataloader(test, len(test_idx))
    model = MyGIN(in_feat=args.in_feats, out_feat=args.hidden_feat, num_classes=2, p=0.1, num_layers=args.num_layers)
    model = model.to(args.device) 
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    #loss_criterion = nn.CrossEntropyLoss(weight=weights)
    #loss_criterion = nn.CrossEntropyLoss()
    loss_criterion = FocalLoss(gamma=3, alpha=0.4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    
    for epoch in tqdm(range(args.n_epoch)):
        # step 1: train
        model.train()
        train_loss_t, train_acc_t, train_acc_sp_t, train_acc_sn_t = train_func(args, epoch, train_dataloader, model, loss_criterion, optimizer=optimizer)
        
        train_running_loss_s[i].append(train_loss_t)
        train_running_acc_s[i].append(train_acc_t)
        train_running_sp_acc_s[i].append(train_acc_sp_t)
        train_running_sn_acc_s[i].append(train_acc_sn_t)
        model.eval()
        # step 2: evaluation
        test_loss_t, test_acc_t, test_sp_acc_t, test_sn_acc_t,  = evaluation(args, epoch, test_dataloader, model, loss_criterion)
        test_running_loss_s[i].append(test_loss_t)
        test_running_acc_s[i].append(test_acc_t)
        test_running_sp_acc_s[i].append(test_sp_acc_t)
        test_running_sn_acc_s[i].append(test_sn_acc_t)

        # step 3: scheduler
        scheduler.step(test_loss_t)
    print("{} th train_running_loss = {}, train_running_acc = {}, \n train_running_sp_acc = {}, train_running_sn_acc = {}".format(i, train_running_loss_s[i][-1], train_running_acc_s[i][-1], train_running_sp_acc_s[i][-1],train_running_sn_acc_s[i][-1]) )  
    print("%d th test_running_loss = %f, test_running_acc = %f, \n test_running_sp_acc=%f, test_running_sn_acc = %f" %(i, test_running_loss_s[i][-1], test_running_acc_s[i][-1], test_running_sp_acc_s[i][-1],test_running_sn_acc_s[i][-1]) )
    
    torch.save(model.state_dict(), "./model/GIN/%d/GIN.pt"%(i + 1))
    i += 1

print("Saving five models")
args.device = 'cpu'

model1 = MyGIN(in_feat=args.in_feats, out_feat=args.hidden_feat, num_classes=2, p=0.2, num_layers=args.num_layers)
model1 = model1.eval().to(args.device)
model1.load_state_dict(torch.load("./model/GIN/1/GIN.pt"))

model2 = MyGIN(in_feat=args.in_feats, out_feat=args.hidden_feat, num_classes=2, p=0.2, num_layers=args.num_layers)
model2 = model2.eval().to(args.device)
model2.load_state_dict(torch.load("./model/GIN/2/GIN.pt"))

model3 = MyGIN(in_feat=args.in_feats, out_feat=args.hidden_feat, num_classes=2, p=0.2, num_layers=args.num_layers)
model3 = model3.eval().to(args.device)
model3.load_state_dict(torch.load("./model/GIN/3/GIN.pt"))

model4 = MyGIN(in_feat=args.in_feats, out_feat=args.hidden_feat, num_classes=2, p=0.2, num_layers=args.num_layers)
model4 = model4.eval().to(args.device)
model4.load_state_dict(torch.load("./model/GIN/4/GIN.pt"))

model5 = MyGIN(in_feat=args.in_feats, out_feat=args.hidden_feat, num_classes=2, p=0.2, num_layers=args.num_layers)
model5 = model5.eval().to(args.device)
model5.load_state_dict(torch.load("./model/GIN/5/GIN.pt"))
models = [model1, model2, model3, model4, model5]


print('\npredicting all test results of each model\n')
# check final test predicted result of five model
test_dataloader = get_dataloader(test_data, len(test_data))
loss_criterion = FocalLoss(gamma=3, alpha=0.4)
logits = []
losses = []
for idx, batch_data in enumerate(test_dataloader):
    # step 1: 准备数据
    smiles, graphs, labels, masks = batch_data
    graph, labels, masks = graphs.to(args.device), labels.long().to(args.device), masks.to(args.device)
    # step 2: 拟合数据
    #logit, _, _ = fit(args, model5, graphs)
    
    for model in models:
        logit = fit(args, model, graphs, smiles)
        logits.append(logit)
        # step 3: 求损失
        loss = loss_criterion(logit, labels.squeeze())
        losses.append(loss.item())
        # step 4: 求准确率
        acc_t, sp_t, sn_t = compute_accuracy(logit, labels)
        print(acc_t, sp_t, sn_t)
acc_t, sp_acc_t, sn_acc_t = vote(logits, labels, args.device)
print(acc_t, sp_acc_t, sn_acc_t)

print('\npredicting all valid results of each model\n')
valid_dataloader = get_dataloader(valid_data, len(valid_data))
logits = []
losses = []
for idx, batch_data in enumerate(valid_dataloader):
    # step 1: 准备数据
    smiles, graphs, labels, masks = batch_data
    graph, labels, masks = graphs.to(args.device), labels.long().to(args.device), masks.to(args.device)
    # step 2: 拟合数据
    #logit, _, _ = fit(args, model5, graphs)
    
    for model in models:
        logit = fit(args, model, graphs, smiles)
        logits.append(logit)
        # step 3: 求损失
        loss = loss_criterion(logit, labels.squeeze())
        losses.append(loss.item())
        acc_t, sp_t, sn_t = compute_accuracy(logit, labels)
        print(acc_t, sp_t, sn_t)
    # step 4: 求准确率

acc_t, sp_acc_t, sn_acc_t = vote(logits, labels, args.device)
print(acc_t, sp_acc_t, sn_acc_t)