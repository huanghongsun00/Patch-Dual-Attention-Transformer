import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from Settings import seed_torch
from sklearn.metrics import mean_absolute_error, r2_score
import torch.utils.data as Data
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
import argparse
from model.Patch_DAT_model import Patch_DAT_backbone
import warnings
warnings.filterwarnings('ignore')


def data_prepare_array(X, y,info):
    X_new, y_new, y_subject_new, info_new= [], [], [], []
    for i in range(len(y)):
        X_new.append(np.array(X[i]))
        y_new.append(np.array(y[i]))
        info_new.append(np.array(info[i]))
    return np.array(X_new,dtype=object), np.array(y_new,dtype=object), np.array(info_new,dtype=object)

def data_reshape(data):
    data_new = []
    for j in range(len(data)):
        data_new.extend(data[j])
    return np.array(data_new)

def X_crop(X, Window_flag = 0):
    if Window_flag == 0:
        return np.float32(X)
    elif Window_flag == 1:
        return np.float32(X[:, :241])

def reshape_and_scaling3(X_train,X_val,Window_flag=0, Normalization=True, type=0):
    X_train, X_val = data_reshape(X_train), data_reshape(X_val)
    X_train, X_val = X_crop(X_train, Window_flag), X_crop(X_val, Window_flag)
    if Normalization:
        if type == 0:
            max_value = np.max(np.max(X_train))
            min_value = np.min(np.min(X_train))
            X_train = (X_train - min_value) / (max_value - min_value)
            X_val = (X_val - min_value) / (max_value - min_value)
        elif type == 1:
            Scaler = MinMaxScaler()
            Scaler.fit(X_train)
            X_train = Scaler.transform(X_train)
            X_val = Scaler.transform(X_val)
    return X_train, X_val

def calculate_mae_loss(predictions, ground_truths):
    loss = torch.mean(torch.abs(predictions - ground_truths))
    return loss

def get_finaldata():
    path_alldata = "your_data_path/"
    X = np.load(path_alldata + 'your_data.npy', allow_pickle=True)
    y = np.load(path_alldata + 'your_label.npy', allow_pickle=True)
    info = np.load(path_alldata + 'other_information.npy',allow_pickle=True) # [id,date,age,sex,inten,side,order]
    return data_prepare_array(X, y, info)

def label_extract(label):
    return [[[j[0], j[1], j[2]] for j in i] for i in label]

def get_final_result(Results,inforamtion):
    info_all = inforamtion
    info_labels = ['id', 'date', 'age', 'sex', 'inten', 'side', 'order']
    DF = pd.DataFrame(info_all, columns=info_labels)
    DF_ = DF
    DF_['true1'] = pd.Series()
    DF_['pred1'] = pd.Series()
    DF_['true3'] = pd.Series()
    DF_['pred3'] = pd.Series()
    DF_['true5'] = pd.Series()
    DF_['pred5'] = pd.Series()

    merged_df = pd.merge(DF_, Results, on=['id', 'date', 'inten', 'side', 'order'])
    merged_df['true1'] = merged_df['true11']
    merged_df['true3'] = merged_df['true33']
    merged_df['true5'] = merged_df['true55']
    merged_df['pred1'] = merged_df['pred11']
    merged_df['pred3'] = merged_df['pred33']
    merged_df['pred5'] = merged_df['pred55']

    merged_df = merged_df.drop(['true11', 'pred11'], axis=1)
    merged_df = merged_df.drop(['true33', 'pred33'], axis=1)
    merged_df = merged_df.drop(['true55', 'pred55'], axis=1)

    # Initialize counters for different error scale
    a = {0.1: 0, 0.15: 0, 0.2: 0}
    b = {0.1: 0, 0.15: 0, 0.2: 0}
    c = {0.1: 0, 0.15: 0, 0.2: 0}

    # Iterate over each row in the dataframe
    for _, row in merged_df.iterrows():
        # Compute absolute differences between predicted and true values
        a1 = np.abs(row['pred1'] - row['true1'])
        a3 = np.abs(row['pred3'] - row['true3'])
        a5 = np.abs(row['pred5'] - row['true5'])

        # Update counters for each condition
        for threshold in a:
            if a1 <= threshold:
                a[threshold] += 1
            if a3 <= threshold:
                b[threshold] += 1
            if a5 <= threshold:
                c[threshold] += 1

    # Total number of samples
    len_row = merged_df.shape[0]
    # Print results for each threshold
    # Calculate the combined accuracy for each threshold
    acc = {threshold: ((a[threshold] + b[threshold] + c[threshold]) * 100) / (len_row * 3) for threshold in a}
    # Print the combined accuracy for each threshold
    for threshold in acc:
        print(f"External test: Combined accuracy within {threshold}ms: {acc[threshold]}")

def train_test(args,train_data_loader, val_data_loader,train_loss,val_loss,size, model, optimizer,scheduler):
    train_mae_per_epoch, test_mae_per_epoch = [], []
    for epoch in range(args.num_epochs):
        print('Epoch:', epoch)
        if epoch % 5 == 0:
            print("lr of %d epoch ：%f" % (epoch, optimizer.param_groups[0]['lr']))
        model.train()
        scheduler.step()
        for i, (batch_X, batch_y, batch_info) in enumerate(tqdm(train_data_loader)):
            if torch.cuda.is_available():
                batch_X = batch_X.to(args.device)
                batch_y = batch_y.to(args.device)
            prediction = model.forward(batch_X)
            loss = calculate_mae_loss(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        loss_x = 0
        for i, (batch_X, batch_y, batch_info) in enumerate(tqdm(train_data_loader)):
            if torch.cuda.is_available():
                batch_X = batch_X.to(args.device)
                batch_y = batch_y.to(args.device)

            training_prediction = model.forward(batch_X)
            loss = calculate_mae_loss(training_prediction, batch_y)
            loss_x += loss.item()

            if i == 0:
                y_train_pred = training_prediction.detach().cpu().numpy()
                y_train_true = batch_y.cpu().numpy()
                y_train_info = batch_info.cpu().numpy()
            else:
                y_train_pred = np.concatenate((y_train_pred, training_prediction.detach().cpu().numpy()))
                y_train_true = np.concatenate((y_train_true, batch_y.cpu().numpy()))
                y_train_info = np.concatenate((y_train_info, batch_info.cpu().numpy()))

        train_loss[epoch] = loss_x / size[0]
        train_mae_per_epoch.append(mean_absolute_error(y_train_true, y_train_pred))

        loss_x = 0
        for i, (batch_X, batch_y, batch_info) in enumerate(tqdm(val_data_loader)):
            if torch.cuda.is_available():
                batch_X = batch_X.to(args.device)
                batch_y = batch_y.to(args.device)

            test_prediction = model.forward(batch_X)
            loss = calculate_mae_loss(test_prediction, batch_y)
            loss_x += loss.item()

            if i == 0:
                y_test_pred = test_prediction.detach().cpu().numpy()
                y_test_true = batch_y.cpu().numpy()
                y_test_info = batch_info.cpu().numpy()
            else:
                y_test_pred = np.concatenate((y_test_pred, test_prediction.detach().cpu().numpy()))
                y_test_true = np.concatenate((y_test_true, batch_y.cpu().numpy()))
                y_test_info = np.concatenate((y_test_info, batch_info.cpu().numpy()))

        val_loss[epoch] = loss_x / size[1]
        test_mae_per_epoch.append(mean_absolute_error(y_test_true, y_test_pred))

        if epoch == args.num_epochs - 1:
            indices = [2, 3]
            for i in range(len(y_test_pred)):
                per_df = []
                if i == 0:
                    info_small = np.delete(y_test_info[i], indices, axis=0)
                    id_ = info_small[0]
                    date_ = info_small[1]
                    inten_ = info_small[2]
                    side_ = info_small[3]
                    order_ = info_small[4]
                    per_df.append([id_, date_, inten_, side_, order_,
                                   y_test_true[i][0], y_test_pred[i][0],
                                   y_test_true[i][1], y_test_pred[i][1],
                                   y_test_true[i][2], y_test_pred[i][2]])
                    df_ = pd.DataFrame(per_df,columns=['id', 'date', 'inten', 'side', 'order', 'true11', 'pred11', 'true33','pred33', 'true55', 'pred55'])

                else:
                    info_small = np.delete(y_test_info[i], indices, axis=0)
                    id_ = info_small[0]
                    date_ = info_small[1]
                    inten_ = info_small[2]
                    side_ = info_small[3]
                    order_ = info_small[4]
                    per_df.append([id_, date_, inten_, side_, order_,
                                   y_test_true[i][0], y_test_pred[i][0],
                                   y_test_true[i][1], y_test_pred[i][1],
                                   y_test_true[i][2], y_test_pred[i][2]])
                    df_1 = pd.DataFrame(per_df,columns=['id', 'date', 'inten', 'side', 'order', 'true11', 'pred11', 'true33','pred33', 'true55', 'pred55'])
                    df_ = pd.concat([df_, df_1])
    plt.figure()
    plt.plot(train_mae_per_epoch, c='b', label='training set')
    plt.plot(test_mae_per_epoch, c='r', label='test set')
    plt.xlabel('Iterations')
    plt.ylabel('Mean-absolute-error(dB)')
    plt.legend(loc='best')
    plt.title(f'Patch-DAT external test -mean absolute error')
    plt.show()
    return df_


def main(X_train_ori, X_test, y_train_ori, y_test, info_train_ori, info_test):

    X_train, y_train = X_train_ori, y_train_ori
    y_train, y_test = label_extract(y_train), label_extract(y_test)
    y_train, y_test = data_reshape(y_train), data_reshape(y_test)

    info_train = info_train_ori
    info_train = data_reshape(info_train)
    info_test = data_reshape(info_test)

    X_train, X_test = reshape_and_scaling3(X_train, X_test, Window_flag=args.Window_flag, Normalization=args.Normalization,type=args.normalization_type)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    size = [len(X_train), len(X_test)]
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    info_train_cleaned = np.array([[int(''.join(filter(str.isdigit, item))) for item in row] for row in info_train])
    info_train = np.array(info_train_cleaned).astype(int)
    info_train = torch.from_numpy(info_train)

    info_test_cleaned = np.array([[int(''.join(filter(str.isdigit, item))) for item in row] for row in info_test])
    info_test = np.array(info_test_cleaned).astype(int)
    info_test = torch.from_numpy(info_test)
    info_all = info_test

    train_dataset = Data.TensorDataset(X_train, y_train, info_train)
    train_data_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = Data.TensorDataset(X_test, y_test, info_test)
    test_data_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = Patch_DAT_backbone(c_in=args.c_in, context_window=args.context_window, target_window=args.target_window, patch_len=args.patch_len,
                               stride=args.stride,n_layers=args.n_layers, d_model=args.dim, n_heads=args.num_heads)
    model = model.to(args.device)
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.6f" % (total))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, amsgrad=True, betas=(0.9, 0.999))
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    train_loss, val_loss = np.zeros([args.num_epochs, 1]), np.zeros([args.num_epochs, 1])
    Results = train_test(args,train_data_loader, test_data_loader,train_loss,val_loss,size,  model=model, optimizer=optimizer,scheduler=scheduler)  # Obtain training and testing results
    get_final_result(Results,info_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=6, help='number of Patch-DAT enocer layers')
    parser.add_argument('--patch_len', type=int, default=20, help='length of each patch')
    parser.add_argument('--stride', type=int, default=5, help='Step size of patching')
    parser.add_argument('--dim', type=int, default=64, help='encoding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='num of attention head')
    parser.add_argument('--c_in', type=int, default=1, help='number of data channels')
    parser.add_argument('--context_window', type=int, default=241, help='number of data time points')
    parser.add_argument('--target_window', type=int, default=3, help='number of output')

    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--test_size', type=float, default=0.1,help='dataset partition size')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--Normalization', default=True)
    parser.add_argument('--normalization_type', default=0,help='normalization method, based on the training set')
    parser.add_argument('--learning_rate', type=float,default=1e-3)
    parser.add_argument('--gamma', type=float,default=0.8)
    parser.add_argument('--Window_flag', default=1,help='data cropping size')
    parser.add_argument('--Random_num', default=888)
    parser.add_argument('--device', type=int, default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    args = parser.parse_args()

    seed_torch(args.Random_num)
    X, y, info = get_finaldata()
    X_train_ori, X_test, y_train_ori, y_test, info_train_ori, info_test = train_test_split(X, y, info,test_size=args.test_size,random_state=args.Random_num)
    print(len(data_reshape(X_train_ori)))
    print(len(data_reshape(X_test)))

    main(X_train_ori, X_test, y_train_ori, y_test, info_train_ori, info_test)





