
# encoding: utf-8
import numpy as np
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.utils.data as Data
import torch
from Train.utils import acc_target, set_seed,entropy_minimization_loss
import time
import logging
import os

from Models.HybGNN import HybGNN_IA,HybGNN_Fix

def train(arg,dest):
    set_seed(arg.seed)

    time_start = time.time()
    kf = KFold(n_splits=10, shuffle=True, random_state=arg.seed)
    sub_info = np.arange(53)  # 行索引从1到54，列索引为0（A列）
    npzloader=np.load(arg.dataset_path)
    data=npzloader['data']
    label=npzloader['label']
    _, _, chan_num, tl = data.shape

    batchsize = arg.bs

    learning_rate = arg.lr
    weight_decay = arg.wd
    total_epoch = arg.total_epoch
    miu=arg.miu

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc_max_all, all_max_f1, all_max_pre, all_max_recall = [], [], [], []
    acc_last, f1_last, pre_last, recall_last = [], [], [], []


    for fold, (train_idx, test_idx) in enumerate(kf.split(sub_info)):

        train_data = np.array([data[i] for i in train_idx]).reshape(-1,chan_num,tl)
        train_label = np.array([label[i] for i in train_idx]).reshape(-1)
        test_data = np.array([data[i] for i in test_idx]).reshape(-1, chan_num, tl)
        test_label = np.array([label[i] for i in test_idx]).reshape(-1)

        print(fold,train_data.shape, train_label.shape, test_data.shape, test_label.shape)

        train_data, train_label, test_data, test_label = map(torch.from_numpy,
                                                                   [train_data, train_label, test_data, test_label])
        train_set = Data.TensorDataset(train_data, train_label)
        test_set = Data.TensorDataset(test_data, test_label)
        train_loader = Data.DataLoader(train_set, batch_size=batchsize, shuffle=True)
        test_loader = Data.DataLoader(test_set, batch_size=batchsize, shuffle=True)

        # Integrate GPUM with FGNN or IAGNN
        if arg.flag=='FGNN':
            model = HybGNN_Fix(input_sample=arg.inputsp,N_next=arg.N_next).to(device)
        if arg.flag=='IAGNN':
            model = HybGNN_IA(input_sample=arg.inputsp,N_next=arg.N_next).to(device)


        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_data_size, test_data_size = len(train_set), len(test_set)
        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []
        val_acc_max, val_f1_max, val_pre_max, val_recall_max = 0.0, 0.0, 0.0, 0.0
        val_acc_max_epoch = 0,
        logging.info('\n\n'+'=' * 20 + f'fold:{fold}' + '=' * 20)


        for epoch in range(total_epoch):

            model.train()

            train_epoch_loss = 0.0
            epoch_train_acc_num = 0
            for i, (features, labels) in enumerate(train_loader):
                features, labels = features.float().to(device), labels.long().to(device)

                optimizer.zero_grad()

                if arg.flag=='IAGNN':
                    outputs, R_data1, R_data2 = model(features)
                else:
                    outputs, R_data1= model(features)

                if miu > 0:
                    loss = criterion(outputs, labels) + entropy_minimization_loss(R_data1,
                                                                                      miu)
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                acc_num = (predicted == labels).sum()
                epoch_train_acc_num += acc_num

            train_epoch_loss /= len(train_loader)
            train_loss_history.append(train_epoch_loss)
            train_acc = epoch_train_acc_num / train_data_size
            train_acc_history.append(train_acc.item())

            model.eval()
            with torch.no_grad():
                True_label = []
                Predict_label = []
                for i, (features, labels) in enumerate(test_loader):
                    features = features.float().to(device)
                    if arg.flag == 'IAGNN':
                        outputs, _,_ = model(features)
                    else:
                        outputs,_ = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    Predict_label.extend(predicted.cpu().tolist())
                    True_label.extend(labels.tolist())
                val_acc, precision, recall, val_f1 = acc_target(Predict_label, True_label)

                if (val_acc > val_acc_max):
                    val_acc_max = val_acc
                    val_pre_max = precision
                    val_recall_max = recall
                    val_f1_max = val_f1
                    val_acc_max_epoch = epoch + 1
                    if (val_acc_max == 1 and val_pre_max == 1 and val_recall_max == 1 and val_f1_max == 1):
                        break
                    logging.info(
                        f'Epoch {epoch + 1}, Train_loss:{train_epoch_loss:.3f}, Train_acc:{train_acc:.3f}, Val_acc:{val_acc:.3f},'
                        f' Best_acc:{val_acc_max:.3f} , Best_f1:{val_f1_max:.3f} , Best_pre:{val_pre_max:.3f} , Best_recall:{val_recall_max:.3f} in[{val_acc_max_epoch}]')

        acc_max_all.append(val_acc_max)
        all_max_pre.append(val_pre_max)
        all_max_recall.append(val_recall_max)
        all_max_f1.append(val_f1_max)
        acc_last.append(val_acc)
        pre_last.append(precision)
        recall_last.append(recall)
        f1_last.append(val_f1)

        modelpth_name='model'+str(fold)+'.pth'
        model_save_path=os.path.join(dest,modelpth_name)
        torch.save(model.state_dict(), model_save_path)

    acc_max_all_array = np.array(acc_max_all)
    pre_max_all_array = np.array(all_max_pre)
    recall_max_all_array = np.array(all_max_recall)
    f1_max_all_array = np.array(all_max_f1)
    acc_last_array = np.array(acc_last)
    pre_last_array = np.array(pre_last)
    recall_last_array = np.array(recall_last)
    f1_last_array = np.array(f1_last)


    time_end = time.time()
    time_sum = time_end - time_start

    logging.info('\n\n'+'=' * 20 + f'Conclusion' + '=' * 20)
    logging.info(acc_max_all_array)
    logging.info('\n')
    logging.info(f'ACC: {np.mean(acc_max_all_array)} , {np.std(acc_max_all_array)}')
    logging.info(f'PRE: {np.mean(pre_max_all_array)} , {np.std(pre_max_all_array)}')
    logging.info(f'REC: {np.mean(recall_max_all_array)} , {np.std(recall_max_all_array)}')
    logging.info(f'F1:  {np.mean(f1_max_all_array)} , {np.std(f1_max_all_array)}')
    logging.info('\n')
    logging.info(f'last_ACC:    {np.mean(acc_last_array)} , {np.std(acc_last_array)}')
    logging.info(f'last_PRE:    {np.mean(pre_last_array)} , {np.std(pre_last_array)}')
    logging.info(f'last_REC:    {np.mean(recall_last_array)} , {np.std(recall_last_array)}')
    logging.info(f'last_F1:    {np.mean(f1_last_array)} , {np.std(f1_last_array)}')
    logging.info(f'Running Time: {time_sum}')

