import pandas as pd
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import numpy as np


def draw_plot(epoch, train_loss_list, train_f1_acc_list, train_auc_acc_list, train_acc_list,
              val_loss_list, val_f1_acc_list, val_auc_acc_list, val_acc_list):
    x = range(0, epoch)
    plt.figure(figsize=(11,28))
    plt.subplot(4, 1, 1)
    plt.plot(x,train_loss_list,color="blue",label="train_loss_list Line",linewidth=2)
    plt.plot(x,val_loss_list,color="orange",label="val_loss_list Line",linewidth=2)
    plt.title("Loss_curve",fontsize=20)
    plt.xlabel(xlabel="Epochs", fontsize=15)
    plt.ylabel(ylabel="Loss", fontsize=15)
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(x,train_auc_acc_list,color="blue",label="train_auc_acc_list Line",linewidth=2)
    plt.plot(x,val_auc_acc_list,color="orange",label="val_auc_acc_list Line",linewidth=2)
    plt.title("AUC_curve",fontsize=20)
    plt.xlabel(xlabel="Epochs", fontsize=15)
    plt.ylabel(ylabel="Accuracy", fontsize=15)
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(x, train_f1_acc_list, color="blue", label="train_f1_acc_list Line", linewidth=2)
    plt.plot(x, val_f1_acc_list, color="orange", label="val_f1_acc_list Line", linewidth=2)
    plt.title("F1_curve", fontsize=20)
    plt.xlabel(xlabel="Epochs", fontsize=15)
    plt.ylabel(ylabel="Accuracy", fontsize=15)
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(x, train_acc_list, color="blue", label="train_f1_acc_list Line", linewidth=2)
    plt.plot(x, val_acc_list, color="orange", label="val_f1_acc_list Line", linewidth=2)
    plt.title("Acc_curve", fontsize=20)
    plt.xlabel(xlabel="Epochs", fontsize=15)
    plt.ylabel(ylabel="Accuracy", fontsize=15)
    plt.legend()
    plt.savefig("Loss&acc.jpg")



def count(data):
    sett = set(data)
    dict = {}
    for item in sett:
        dict.update({item: data.count(item)})
    print(dict)


def accuracy(y, pred):
    result = 0
    lenth = len(y)
    for i in range(lenth):
        if y[i] == pred[i]:
            result += 1
    return result / lenth


def str2lst(i):
    return list(map(float, i.split(" ")))


class MLP(nn.Module):
    def __init__(self, layers, layers_size, dropout, modelDir="model_save", device=torch.device("cpu")):
        super(MLP, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-chinese",
                                                       cache_dir=modelDir)

        self.layers = nn.Sequential().to(device)
        for i in range(layers-1):
            self.layers.add_module("Linear%d" % (i), nn.Linear(layers_size[i], layers_size[i+1]).to(device))
            self.layers.add_module("ReLU%d" % (i), nn.ReLU(inplace=True).to(device))
        self.layers.add_module("Linear%d" % (layers-1), nn.Linear(layers_size[-2], layers_size[-1]).to(device))
        self.device = device

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float).to(self.device)
        output = self.layers(x)
        return output


epoch = 500
device = torch.device("cuda")
batch_size = 1
loss_and_acc_curve = True

test = pd.read_csv("data/test.csv")
test_x = test["embed"].tolist()
test_y = torch.as_tensor(test["difficulty"]).to(torch.float).to(device)
test_x = [str2lst(str(i)) for i in test_x]

train = pd.read_csv("data/train.csv")
train_x = train["embed"].tolist()
train_y = torch.as_tensor(train["difficulty"]).to(torch.float).to(device)
train_x = [str2lst(str(i)) for i in train_x]

mlp = MLP(2, [768, 64, 4], 0.5, device=device)
optimizer = optim.SGD(params=mlp.parameters(), lr=1e-2)
loss_func = nn.CrossEntropyLoss().to(device)
train_loss_list = []
train_f1_acc_list = []
train_auc_acc_list = []
train_acc_list = []
val_loss_list = []
val_f1_acc_list = []
val_auc_acc_list = []
val_acc_list = []

for i in range(epoch):
    best_acc = 0

    train_loss = 0
    val_loss = 0
    total_loss = 0
    total_val_loss = 0

    epoch_size_trn = train_y.size(0) // batch_size
    epoch_size_val = test_y.size(0) // batch_size

    sm = nn.Softmax(dim=1).to(device)

    train_target = None
    val_target = None
    train_f1 = None
    train_auc = None
    val_f1 = None
    val_auc = None

    # mlp.train()
    for i in tqdm(range(len(train_y))):
        x = train_x[i]
        y = train_y[i].long()
        optimizer.zero_grad()
        if len(x) != 768:
            continue
        pred = mlp(x)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        unpred = pred.unsqueeze(dim=0)
        argpred = torch.argmax(unpred, dim=1)
        softpred = sm(unpred)

        if train_target is None:
            train_target = y.unsqueeze(dim=0)
        else:
            train_target = torch.cat((train_target, y.unsqueeze(dim=0)), dim=0)

        if train_f1 is None:
            train_f1 = argpred
        else:
            train_f1 = torch.cat((train_f1, argpred), dim=0)

        if train_auc is None:
            train_auc = softpred
        else:
            train_auc = torch.cat((train_auc, softpred), dim=0)

    train_loss = total_loss / epoch_size_trn
    train_f1_acc = f1_score(train_target.detach().cpu().numpy(), train_f1.detach().cpu().numpy(), average='micro')
    train_auc_acc = roc_auc_score(train_target.detach().cpu().numpy(), train_auc.detach().cpu().numpy(), multi_class='ovr')
    train_acc = accuracy(train_target.detach().cpu().numpy(), train_f1.detach().cpu().numpy())

    # mlp.eval()
    for i in range(len(test_y)):
        x = test_x[i]
        y = test_y[i].long()
        optimizer.zero_grad()
        if len(x) != 768:
            continue
        pred = mlp(x)
        loss = loss_func(pred, y)

        total_val_loss += loss.item()

        argpred = torch.argmax(pred, dim=1)
        softpred = sm(pred) # 加 softmax 给 auc

        if val_target is None:
            val_target = y
        else:
            val_target = torch.cat((val_target, y), dim=0)

        if val_f1 is None:
            val_f1 = argpred
        else:
            val_f1 = torch.cat((val_f1, argpred), dim=0)

        if val_auc is None:
            val_auc = softpred
        else:
            val_auc = torch.cat((val_auc, softpred), dim=0)

    val_loss = total_val_loss / epoch_size_val
    val_f1_acc = f1_score(val_target.detach().cpu().numpy(), val_f1.detach().cpu().numpy(), average='micro')
    val_auc_acc = roc_auc_score(val_target.detach().cpu().numpy(), val_auc.detach().cpu().numpy(), multi_class='ovr')
    val_acc = accuracy(val_target.detach().cpu().numpy(), val_f1.detach().cpu().numpy())

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(mlp.state_dict(), "weight/mlp_weight-epoch%d.pt" % (i))

    train_loss_list.append(train_loss)
    train_f1_acc_list.append(train_f1_acc)
    train_auc_acc_list.append(train_auc_acc)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_f1_acc_list.append(val_f1_acc)
    val_auc_acc_list.append(val_auc_acc)
    val_acc_list.append(val_acc)


if loss_and_acc_curve:
    draw_plot(epoch, train_loss_list, train_f1_acc_list, train_auc_acc_list, train_acc_list,
              val_loss_list, val_f1_acc_list, val_auc_acc_list, val_acc_list)
